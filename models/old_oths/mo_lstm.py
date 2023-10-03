# use nn.LSTM, no content, only [f01, ..., f44]

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


class PC_MO_LSTM(nn.Module):
    def __init__(self, use_knn=True):
        super(PC_MO_LSTM, self).__init__()
        self.sa1 = SAModule(float(1/16), MLP([3+3, 32, 32, 64],norm='batch_norm'), r=0.2*8.5, k=32, knn=use_knn)
        self.sa2 = SAModule(0.5, MLP([64+3, 96, 96, 128],norm='batch_norm'), r=0.4*8.5, k=16, knn=use_knn)
        self.sa3 = SAModule(0.5, MLP([128+3, 128, 128, 256],norm='batch_norm'), r=0.8*8.5, k=8, knn=use_knn)

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256

        self.local_gat1 = Local_Point_Trans(32, self.C1 , self.C1)
        self.local_gat2 = Local_Point_Trans(16, self.C2 , self.C2)
        self.local_gat3 = Local_Point_Trans(8, self.C3 , self.C3)

        self.lstm_1 = nn.LSTM(input_size=self.C1, hidden_size=2*self.C1, num_layers=2, batch_first=False)
        self.lstm_2 = nn.LSTM(input_size=self.C2, hidden_size=2*self.C2, num_layers=2, batch_first=False)
        self.lstm_3 = nn.LSTM(input_size=self.C3, hidden_size=2*self.C3, num_layers=2, batch_first=False)

        self.fp32 = FPModule(8, MLP([768+384, 768, 576], norm='batch_norm'))
        self.fp21 = FPModule(16, MLP([576+192, 576, 384],norm='batch_norm'))
        self.fp10 = FPModule(32, MLP([384, 256, 128], norm='batch_norm'))

        self.classifier1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1, bias=False)
        self.classifier4 = nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1, bias=False)

    
    def forward(self, input_xyz_list, num_pred, device):

        total_l = len(input_xyz_list) + num_pred
        
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
        last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
        n1, h1 = f1_list[-1].shape
        n2, h2 = f2_list[-1].shape
        n3, h3 = f3_list[-1].shape

        ft1_list = torch.empty((len(f1_list)-1, n1, h1)).to(device)
        ft2_list = torch.empty((len(f2_list)-1, n2, h2)).to(device)
        ft3_list = torch.empty((len(f3_list)-1, n3, h3)).to(device)

        for i in range(len(f1_list)):
            if i < len(f1_list)-1:
                ft1_list[i] = self.local_gat1(f1_list[i+1], f1_list[i], xyz1_list[i+1], xyz1_list[i], batch1,t_i=float(i+1/total_l), t_last=float((i)/total_l))
                ft2_list[i] = self.local_gat2(f2_list[i+1], f2_list[i], xyz2_list[i+1], xyz2_list[i], batch2,t_i=float(i+1/total_l), t_last=float((i)/total_l))
                ft3_list[i] = self.local_gat3(f3_list[i+1], f3_list[i], xyz3_list[i+1], xyz3_list[i], batch3,t_i=float(i+1/total_l), t_last=float((i)/total_l))
            else:
                ftt1 = self.local_gat1(f1_list[i], f1_list[i], xyz1_list[i], xyz1_list[i], batch1,t_i=float((i)/total_l), t_last=float((i)/total_l))
                ftt2 = self.local_gat2(f2_list[i], f2_list[i], xyz2_list[i], xyz2_list[i], batch2,t_i=float((i)/total_l), t_last=float((i)/total_l))
                ftt3 = self.local_gat3(f3_list[i], f3_list[i], xyz3_list[i], xyz3_list[i], batch3,t_i=float((i)/total_l), t_last=float((i)/total_l))
        
        # [f01, f12, f23, f34]

        f45_3, _ = self.lstm_3(ft3_list)
        tm3 = torch.cat([f45_3[-1], ftt3], dim=-1) #(B*N/32, 768)
        f45_2, _ = self.lstm_2(ft2_list)
        tm2 = torch.cat([f45_2[-1], ftt2], dim=-1) #(B*N/16, 384)
        f45_1, _ = self.lstm_1(ft1_list)
        tm1 = torch.cat([f45_1[-1], ftt1], dim=-1) #(B*N/4, 192)
        
        x2, pos_2, batch_2 = self.fp32(tm3, last_xyz3, batch3, tm2, last_xyz2, batch2)
        x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, tm1, last_xyz1, batch1)
        x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)

        pc_next = last_in_xyz + self.classifier4(self.classifier3(self.classifier2(self.classifier1(x0.T)))).T
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
            out1 = self.local_gat1(f1_list[-1], f1_list[-2], xyz1_list[-1], xyz1_list[-2], batch1,t_i=float((l)/total_l), t_last=float((l-1)/total_l))
            out2 = self.local_gat2(f2_list[-1], f2_list[-2], xyz2_list[-1], xyz2_list[-2], batch2,t_i=float((l)/total_l), t_last=float((l-1)/total_l))
            out3 = self.local_gat3(f3_list[-1], f3_list[-2], xyz3_list[-1], xyz3_list[-2], batch3,t_i=float((l)/total_l), t_last=float((l-1)/total_l))
            
            ft1_list = torch.cat([ft1_list, out1.unsqueeze(0)], dim=0)
            ft2_list = torch.cat([ft2_list, out2.unsqueeze(0)], dim=0)
            ft3_list = torch.cat([ft3_list, out3.unsqueeze(0)], dim=0)

            ftt1 = self.local_gat1(f1_list[-1], f1_list[-1], xyz1_list[-1], xyz1_list[-1], batch1,t_i=float((l)/total_l), t_last=float((l)/total_l))
            ftt2 = self.local_gat2(f2_list[-1], f2_list[-1], xyz2_list[-1], xyz2_list[-1], batch2,t_i=float((l)/total_l), t_last=float((l)/total_l))
            ftt3 = self.local_gat3(f3_list[-1], f3_list[-1], xyz3_list[-1], xyz3_list[-1], batch3,t_i=float((l)/total_l), t_last=float((l)/total_l))
            
            f45_3, _ = self.lstm_3(ft3_list)
            tm3 = torch.cat([f45_3[-1], ftt3], dim=-1) #(B*N/32, 768)
            f45_2, _ = self.lstm_2(ft2_list)
            tm2 = torch.cat([f45_2[-1], ftt2], dim=-1) #(B*N/16, 384)
            f45_1, _ = self.lstm_1(ft1_list)
            tm1 = torch.cat([f45_1[-1], ftt1], dim=-1) #(B*N/4, 192)
            
            x2, pos_2, batch_2 = self.fp32(tm3, last_xyz3, batch3, tm2, last_xyz2, batch2)
            x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, tm1, last_xyz1, batch1)
            x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)

            pc_next = last_in_xyz + self.classifier4(self.classifier3(self.classifier2(self.classifier1(x0.T)))).T

            pred_detail_xyz_list.append(pc_next)
            

        return pred_detail_xyz_list
    

    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device",device)
    data_config = {'root': "/home/stud/ding/PC_FC/PC_forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00']}
    demo_dataset = KittiDataset_2(root=data_config['root'], npoints=16384, input_num=5, pred_num=5, seqs=['00'])
    train_dataloader = DataLoader(demo_dataset, batch_size=2) 
    bin_xyz, _ = next(iter(train_dataloader))
    bin_xyz = [i.to(device) for i in bin_xyz]


    #  input_xyz_list, num_pred, device

    testmodel = PC_MO_LSTM().to(device)
    res = testmodel(bin_xyz, 5, device)
    for i in res:
        print("i.shape",i.shape)
        










        




            








        
