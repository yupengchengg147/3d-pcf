#old mha


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


class MHA_OLD(nn.Module):
    def __init__(self, use_knn=True):
        super(MHA_OLD, self).__init__()
        self.sa1 = SAModule(float(1/32), MLP([3+3, 32, 32, 64],norm='batch_norm'), r=0.2*8.5, k=32, knn=use_knn)
        self.sa2 = SAModule(0.5, MLP([64+3, 96, 96, 128],norm='batch_norm'), r=0.4*8.5, k=16, knn=use_knn)
        self.sa3 = SAModule(0.5, MLP([128+3, 128, 128, 256],norm='batch_norm'), r=0.8*8.5, k=8, knn=use_knn)

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.num_heads = 4

        self.local_gat1 = LPT_BD_not(32, self.C1 , self.C1)
        self.local_gat2 = LPT_BD_not(16, self.C2 , self.C2)
        self.local_gat3 = LPT_BD_not(8, self.C3 , self.C3)

        
        self.seq_att1 = nn.MultiheadAttention(self.C1, self.num_heads)
        self.seq_att2 = nn.MultiheadAttention(self.C2, self.num_heads)
        self.seq_att3 = nn.MultiheadAttention(self.C3, self.num_heads)


        self.fp32 = FPModule(8, MLP([(256+128)*2, 256*2, 256, 256], norm='batch_norm'))
        self.fp21 = FPModule(16, MLP([256+64*2, 256, 256],norm='batch_norm'))
        self.fp10 = FPModule(32, MLP([256, 256, 256], norm='batch_norm'))

        self.classifier1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=False)


        # self.refiner = Refiner(**refiner_args)
    
    def forward(self, input_xyz_list, num_pred, device):
        
        pred_detail_xyz_list = []

        batch_size, df, N = input_xyz_list[0].shape
        t = torch.LongTensor(range(batch_size)).to(device)
        in_batch = torch.repeat_interleave(t, N).to(device)
        
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
            # print("fea1.shape",fea1.shape), print("xyz1.shape",xyz1.shape), print("batch1.shape",batch1.shape)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            # print("fea2.shape",fea2.shape), print("xyz2.shape",xyz2.shape), print("batch2.shape",batch2.shape)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
            f3_list.append(fea3)
            xyz3_list.append(xyz3)
            # print("fea3.shape",fea3.shape), print("xyz3.shape",xyz3.shape), print("batch3.shape",batch3.shape)
        
        ## seq attention
        last_in_xyz = (input_xyz_list[-1].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
        
        pc_next = self.pred_next(last_in_xyz,f1_list, f2_list,f3_list, xyz1_list, xyz2_list, xyz3_list, batch1, batch2, batch3, in_batch, device)
        # print("pc_next.shape",pc_next.shape)
        pred_detail_xyz_list.append(pc_next)

        for i in range(num_pred-1):
            # xyz0 = pred_detail_xyz_list[-1]
            fea1, xyz1, batch1 = self.sa1(pc_next, pc_next, in_batch)
            f1_list.append(fea1)
            xyz1_list.append(xyz1)
            # print("fea1.shape",fea1.shape), print("xyz1.shape",xyz1.shape), print("batch1.shape",batch1.shape)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            # print("fea2.shape",fea2.shape), print("xyz2.shape",xyz2.shape), print("batch2.shape",batch2.shape)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
            f3_list.append(fea3)
            xyz3_list.append(xyz3)
            # print("fea3.shape",fea3.shape), print("xyz3.shape",xyz3.shape), print("batch3.shape",batch3.shape)
            pc_next = self.pred_next(pc_next,f1_list, f2_list,f3_list, xyz1_list, xyz2_list, xyz3_list, batch1, batch2, batch3, in_batch, device)
            pred_detail_xyz_list.append(pc_next)
            # print("pc_next.shape",pc_next.shape)


        return pred_detail_xyz_list
    
    def pred_next(self, last_in_xyz,f1_list, f2_list,f3_list, xyz1_list, xyz2_list, xyz3_list, batch1, batch2, batch3, in_batch, device):
         ## seq attention
        # last_in_xyz = (input_xyz_list[-1].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
        last_f1, last_f2, last_f3 = f1_list[-1], f2_list[-1], f3_list[-1]
        last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]

        n1, h1 = last_f1.shape
        n2, h2 = last_f2.shape
        n3, h3 = last_f3.shape

        ft1_list = torch.empty((len(f1_list)-1, n1, h1)).to(device)
        ft2_list = torch.empty((len(f2_list)-1, n2, h2)).to(device)
        ft3_list = torch.empty((len(f3_list)-1, n3, h3)).to(device)

        for i in range(len(f1_list)-1):

            ft1_list[i] = self.local_gat1(f1_list[i], last_f1, xyz1_list[i], last_xyz1, batch1,t_i=float(i/len(f1_list)), t_last=1.0)

            ft2_list[i] = self.local_gat2(f2_list[i], last_f2, xyz2_list[i], last_xyz2, batch2,t_i=float(i/len(f1_list)), t_last=1.0)
            ft3_list[i] = self.local_gat3(f3_list[i], last_f3, xyz3_list[i], last_xyz3, batch3,t_i=float(i/len(f1_list)), t_last=1.0)
        
        tm1, _ = self.seq_att1(ft1_list[-1].unsqueeze(0), ft1_list, ft1_list)
        tm2, _ = self.seq_att2(ft2_list[-1].unsqueeze(0), ft2_list, ft2_list)
        tm3, _ = self.seq_att3(ft3_list[-1].unsqueeze(0), ft3_list, ft3_list)

        tm1 = tm1.squeeze(0)
        tm2 = tm2.squeeze(0)
        tm3 = tm3.squeeze(0)

        tm1 = torch.cat((tm1, last_f1), dim=-1)
        tm2 = torch.cat((tm2, last_f2), dim=-1)
        tm3 = torch.cat((tm3, last_f3), dim=-1)

        # print("tm1.shape",tm1.shape), #print("tm2.shape",tm2.shape), #print("tm3.shape",tm3.shape)

        x2, pos_2, batch_2 = self.fp32(tm3, last_xyz3, batch3, tm2, last_xyz2, batch2)
        # #print("x2.shape",x2.shape), #print("pos_2.shape",pos_2.shape), #print("batch_2.shape",batch_2.shape)
        x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, tm1, last_xyz1, batch1)
        # #print("x1.shape",x1.shape), #print("pos_1.shape",pos_1.shape), #print("batch_1.shape",batch_1.shape)
        x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)
        # #print("x0.shape",x0.shape), #print("pos_0.shape",pos_0.shape), #print("batch_0.shape",batch_0.shape)

        pc_next = last_in_xyz + self.classifier3(self.classifier2(self.classifier1(x0.T))).T
        # #print("pc_next.shape",pc_next.shape)

        return pc_next
    
        


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config = {'root': "/usr/stud/ype/storage/user/kittidata/dataset/sequences", 'npoints': 16384, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00']}
    demo_dataset = KittiDataset_2(root=data_config['root'], npoints=16384, input_num=5, pred_num=5, seqs=['00'])
    train_dataloader = DataLoader(demo_dataset, batch_size=2) 
    bin_xyz, _ = next(iter(train_dataloader))
    
    bin_xyz = [i.to(device) for i in bin_xyz]
    
    # refiner_kargs = {"in_channels": 256,  "out_channels": 3, "k": 30,  "dilations": (1,2), "add_points": True}
    # att_args, use_knn=True, local_fea_dim = 256
    att_args = {'num_heads': 4, 'num_neighs': 20}

    #  input_xyz_list, num_pred, device

    testmodel = MHA_OLD(att_args).to(device)

    res = testmodel(bin_xyz, 5, device)
    for i in res:
        print("i.shape",i.shape)
        










        




            








        
