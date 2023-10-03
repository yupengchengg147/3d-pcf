import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from dataset.kitti_dataset_v2 import KittiDataset_2
from torch.utils.data import DataLoader
from torch_geometric.nn import MLP
from models.utils import *
from pugcn_lib.models import Refiner


class PC_MO_cons(nn.Module):
    def __init__(self, use_knn=True):
        super(PC_MO_cons, self).__init__()
        self.sa1 = SAModule(float(1/32), MLP([3+3, 32, 32, 64],norm='batch_norm'), r=0.2*8.5, k=16, knn=use_knn)
        self.sa2 = SAModule(0.5, MLP([64+3, 96, 96, 128],norm='batch_norm'), r=0.4*8.5, k=16, knn=use_knn)
        self.sa3 = SAModule(0.5, MLP([128+3, 128, 128, 256],norm='batch_norm'), r=0.8*8.5, k=8, knn=use_knn)

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.num_heads = 4

        self.local_gat1 = Local_Point_Trans(64, self.C1 , self.C1)
        self.local_gat2 = Local_Point_Trans(32, self.C2 , self.C2)
        self.local_gat3 = Local_Point_Trans(16, self.C3 , self.C3)

        self.seq_att1 = nn.MultiheadAttention(self.C1, self.num_heads)
        self.seq_att2 = nn.MultiheadAttention(self.C2, self.num_heads)
        self.seq_att3 = nn.MultiheadAttention(self.C3, self.num_heads)

        self.fp32 = FPModule(16, MLP([256+128, 256, 256], norm='batch_norm'))
        self.fp21 = FPModule(32, MLP([256+64, 256, 256],norm='batch_norm'))
        self.fp10 = FPModule(64, MLP([256, 256, 256], norm='batch_norm'))

        self.classifier1 = nn.Conv1d(in_channels=self.C3, out_channels=128, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=False)
    
    def forward(self, input_xyz_list, num_pred, device):
        
        pred_detail_xyz_list = []
        batch_size, df, N = input_xyz_list[0].shape
        in_batch = torch.repeat_interleave(torch.LongTensor(range(batch_size)), N).to(device)
        
        f1_list = []
        xyz1_list = []
        f2_list = []
        xyz2_list = []
        f3_list = []
        xyz3_list = []

        ## encoder
        for i in range(len(input_xyz_list)):
            xyz0 = (input_xyz_list[i].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
            fea1, xyz1, batch1 = self.sa1(xyz0, xyz0, in_batch)
            f1_list.append(fea1)
            xyz1_list.append(xyz1)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
            f3_list.append(fea3)
            xyz3_list.append(xyz3)

        last_in_xyz = (input_xyz_list[-1].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
        pc_next, ft1_list, ft2_list, ft3_list = self.init_pred_next(last_in_xyz,f1_list, f2_list,f3_list, xyz1_list, xyz2_list, xyz3_list, batch1, batch2, batch3, in_batch, device)
        pred_detail_xyz_list.append(pc_next)

        for i in range(num_pred-1):
            
            fea1, xyz1, batch1 = self.sa1(pc_next, pc_next, in_batch)
            f1_list.append(fea1)
            xyz1_list.append(xyz1)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
            f3_list.append(fea3)
            xyz3_list.append(xyz3)

            last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
            l = len(xyz1_list)
            out1 = self.local_gat1(f1_list[-2], f1_list[-1], xyz1_list[-2], xyz1_list[-1], batch1,t_i=float((l-1)/l), t_last=1.0)
            out2 = self.local_gat2(f2_list[-2], f2_list[-1], xyz2_list[-2], xyz2_list[-1], batch2,t_i=float((l-1)/l), t_last=1.0)
            out3 = self.local_gat3(f3_list[-2], f3_list[-1], xyz3_list[-2], xyz3_list[-1], batch3,t_i=float((l-1)/l), t_last=1.0)
            ft1_list = torch.cat([ft1_list[:-1], out1.unsqueeze(0)], dim=0)
            ft2_list = torch.cat([ft2_list[:-1], out2.unsqueeze(0)], dim=0)
            ft3_list = torch.cat([ft3_list[:-1], out3.unsqueeze(0)], dim=0)

            ft1_list = torch.cat([ft1_list,
                                self.local_gat1(f1_list[-1], f1_list[-1], xyz1_list[-1], xyz1_list[-1], batch1,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
            ft2_list = torch.cat([ft2_list,
                                self.local_gat2(f2_list[-1], f2_list[-1], xyz2_list[-1], xyz2_list[-1], batch2,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
            ft3_list = torch.cat([ft3_list,
                                self.local_gat3(f3_list[-1], f3_list[-1], xyz3_list[-1], xyz3_list[-1], batch3,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
            
            tm1, _ = self.seq_att1(ft1_list[-1].unsqueeze(0), ft1_list, ft1_list)
            tm2, _ = self.seq_att2(ft2_list[-1].unsqueeze(0), ft2_list, ft2_list)
            tm3, _ = self.seq_att3(ft3_list[-1].unsqueeze(0), ft3_list, ft3_list)
            tm1 = tm1.squeeze(0)
            tm2 = tm2.squeeze(0)
            tm3 = tm3.squeeze(0)
            x2, pos_2, batch_2 = self.fp32(tm3, last_xyz3, batch3, tm2, last_xyz2, batch2)
            x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, tm1, last_xyz1, batch1)
            x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, pc_next, in_batch)
            pc_next = pc_next + self.classifier3(self.classifier2(self.classifier1(x0.T))).T
            pred_detail_xyz_list.append(pc_next)

        return pred_detail_xyz_list
    
    def init_pred_next(self, last_in_xyz,f1_list, f2_list,f3_list, xyz1_list, xyz2_list, xyz3_list, batch1, batch2, batch3, in_batch, device):

        last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
        n1, h1 = f1_list[-1].shape
        n2, h2 = f2_list[-1].shape
        n3, h3 = f3_list[-1].shape

        ft1_list = torch.empty((len(f1_list), n1, h1)).to(device)
        ft2_list = torch.empty((len(f2_list), n2, h2)).to(device)
        ft3_list = torch.empty((len(f3_list), n3, h3)).to(device)

        for i in range(len(f1_list)):
            if i < len(f1_list)-1:
                ft1_list[i] = self.local_gat1(f1_list[i], f1_list[i+1], xyz1_list[i], xyz1_list[i+1], batch1,t_i=float(i/len(f1_list)), t_last=float((i+1)/len(f1_list)))
                ft2_list[i] = self.local_gat2(f2_list[i], f2_list[i+1], xyz2_list[i], xyz2_list[i+1], batch2,t_i=float(i/len(f1_list)), t_last=float((i+1)/len(f1_list)))
                ft3_list[i] = self.local_gat3(f3_list[i], f3_list[i+1], xyz3_list[i], xyz3_list[i+1], batch3,t_i=float(i/len(f1_list)), t_last=float((i+1)/len(f1_list)))
            else:
                ft1_list = torch.cat([ft1_list[:-1], self.local_gat1(f1_list[i], f1_list[i], xyz1_list[i], xyz1_list[i], batch1,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
                ft2_list = torch.cat([ft2_list[:-1], self.local_gat2(f2_list[i], f2_list[i], xyz2_list[i], xyz2_list[i], batch2,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
                ft3_list = torch.cat([ft3_list[:-1],  self.local_gat3(f3_list[i], f3_list[i], xyz3_list[i], xyz3_list[i], batch3,t_i=1.0, t_last=1.0).unsqueeze(0)], dim=0)
        
        # [f01, f12, f23, f34, f44] -- 1,2,3
        
        tm1, _ = self.seq_att1(ft1_list[-1].unsqueeze(0), ft1_list, ft1_list)
        tm2, _ = self.seq_att2(ft2_list[-1].unsqueeze(0), ft2_list, ft2_list)
        tm3, _ = self.seq_att3(ft3_list[-1].unsqueeze(0), ft3_list, ft3_list)

        tm1 = tm1.squeeze(0)
        tm2 = tm2.squeeze(0)
        tm3 = tm3.squeeze(0)

        x2, pos_2, batch_2 = self.fp32(tm3, last_xyz3, batch3, tm2, last_xyz2, batch2)
        x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, tm1, last_xyz1, batch1)
        x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)

        pc_next = last_in_xyz + self.classifier3(self.classifier2(self.classifier1(x0.T))).T
        return pc_next, ft1_list, ft2_list, ft3_list
    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config = {'root': "/home/stud/ding/PC_FC/PC_forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00']}
    demo_dataset = KittiDataset_2(root=data_config['root'], npoints=4096, input_num=5, pred_num=5, seqs=['00'])
    train_dataloader = DataLoader(demo_dataset, batch_size=2) 
    bin_xyz, _, _ = next(iter(train_dataloader))
    
    bin_xyz = [i.to(device) for i in bin_xyz]
    
    # refiner_kargs = {"in_channels": 256,  "out_channels": 3, "k": 30,  "dilations": (1,2), "add_points": True}
    # att_args, use_knn=True, local_fea_dim = 256
    att_args = {'num_heads': 4, 'num_neighs': 20}

    #  input_xyz_list, num_pred, device

    testmodel = PC_MO_KNN(att_args).to(device)

    res = testmodel(bin_xyz, 5, device)
    for i in res:
        print("i.shape",i.shape)
        










        




            








        
