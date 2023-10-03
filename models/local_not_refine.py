# 3行都有，bidirectional, local gat没有t
# 采样点数统一用half
# with refiner

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
gt_dir = os.path.abspath(os.path.join(root_dir,'..'))
sys.path.append(root_dir)
sys.path.append(gt_dir)


import torch
from dataset.kitti_dataset_v2 import KittiDataset_2
from torch.utils.data import DataLoader
from torch_geometric.nn import MLP
from models.utils import *

from pugcn_lib.models import Refiner

class PC_MO_NOT_R(nn.Module):
    def __init__(self, use_knn=True):
        super(PC_MO_NOT_R, self).__init__()

        self.te = 16
        
        # self.scale = nn.parameter.Parameter(torch.FloatTensor([8.5]), requires_grad=True)
        self.tde = TD_Encoder(self.te)

        self.sa1 = SAModule(float(1/32), MLP([3+3, 32, 32, 64],norm='batch_norm'), r=0.2*8.5, k=16, knn=use_knn)  #512 * 64
        self.sa2 = SAModule(float(1/2), MLP([64+3, 96, 96, 128],norm='batch_norm'), r=0.4*8.5, k=8, knn=use_knn)  #256 * 128
        self.sa3 = SAModule(float(1/2), MLP([128+3, 128, 128, 256],norm='batch_norm'), r=0.8*8.5, k=8, knn=use_knn) #128 * 256

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256

        self.local_gat1 = LPT_BD_not(16, self.C1 , self.C1)
        self.local_gat2 = LPT_BD_not(16, self.C2 , self.C2)
        self.local_gat3 = LPT_BD_not(8, self.C3 , self.C3)

        self.lstm_1 = MS_LSTM_Cell(self.C1, self.C1, hidden_size= 2*self.C1)
        self.lstm_2 = MS_LSTM_Cell(self.C2, self.C2, hidden_size= 2*self.C2)
        self.lstm_3 = MS_LSTM_Cell(self.C3, self.C3, hidden_size= 2*self.C3)


        self.fp32 = FPModule(8, MLP([512+256, 512, 384], norm='batch_norm'))
        self.fp21 = FPModule(16, MLP([384+128, 384, 256],norm='batch_norm'))
        self.fp10 = FPModule(32, MLP([256, 256, 128], norm='batch_norm'))

        self.refiner = Refiner(in_channels=128, out_channels=3, k=16, dilations=[1,2], add_points=True)
        

    
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
            
            # print("fea1.shape",fea1.shape)

            f1_list.append(fea1)
            xyz1_list.append(xyz1)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)

            # print("fea2.shape",fea2.shape)

            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)

            # print("fea3.shape",fea3.shape)

            f3_list.append(fea3)
            xyz3_list.append(xyz3)

        n1, h1 = f1_list[-1].shape
        n2, h2 = f2_list[-1].shape
        n3, h3 = f3_list[-1].shape

        C0_1 = torch.zeros(size=(n1, 2*self.C1), dtype=torch.float32, device = device)
        C0_2 = torch.zeros(size=(n2, 2*self.C2), dtype=torch.float32, device = device)
        C0_3 = torch.zeros(size=(n3, 2*self.C3), dtype=torch.float32, device = device)
        H0_1 = torch.zeros(size=(n1, 2*self.C1), dtype=torch.float32, device = device)
        H0_2 = torch.zeros(size=(n2, 2*self.C2), dtype=torch.float32, device = device)
        H0_3 = torch.zeros(size=(n3, 2*self.C3), dtype=torch.float32, device = device)

        # [f0, f01, f00]
        f_back_1, f_forward_1  = self.local_gat1(f1_list[0], f1_list[0], f1_list[1], 
                                        xyz1_list[0], xyz1_list[0], xyz1_list[1],
                                        batch1)
        f_back_2, f_forward_2 = self.local_gat2(f2_list[0], f2_list[0], f2_list[1],
                                        xyz2_list[0], xyz2_list[0], xyz2_list[1],
                                        batch2)
        f_back_3, f_forward_3 = self.local_gat3(f3_list[0], f3_list[0], f3_list[1],
                                        xyz3_list[0], xyz3_list[0], xyz3_list[1],
                                        batch3)
        
        H1_1, C1_1 = self.lstm_1(H0_1, C0_1, f_back_1, f_forward_1, f1_list[i])
        H1_2, C1_2 = self.lstm_2(H0_2, C0_2, f_back_2, f_forward_2, f2_list[i])
        H1_3, C1_3 = self.lstm_3(H0_3, C0_3, f_back_3, f_forward_3, f3_list[i])

        for i in range(1,len(f1_list)):
            if i < len(f1_list)-2:
                # [f1, f12, f10], [f2, f23, f21]
                f_back_1, f_forward_1 = self.local_gat1(f1_list[i-1], f1_list[i], f1_list[i+1], 
                                            xyz1_list[i-1], xyz1_list[i], xyz1_list[i+1],
                                            batch1)
                f_back_2, f_forward_2 = self.local_gat2(f2_list[i-1], f2_list[i], f2_list[i+1],
                                            xyz2_list[i-1], xyz2_list[i], xyz2_list[i+1],
                                            batch2)
                f_back_3, f_forward_3 = self.local_gat3(f3_list[i-1], f3_list[i], f3_list[i+1],
                                            xyz3_list[i-1], xyz3_list[i], xyz3_list[i+1],
                                            batch3)
                H1_1, C1_1 = self.lstm_1(H1_1, C1_1, f_back_1, f_forward_1, f1_list[i])
                H1_2, C1_2 = self.lstm_2(H1_2, C1_2, f_back_2, f_forward_2, f2_list[i])
                H1_3, C1_3 = self.lstm_3(H1_3, C1_3, f_back_3, f_forward_3, f3_list[i])
            elif i == len(f1_list)-2:

                #[f3, f34, f32]
                f_back_1, f_forward_1 = self.local_gat1(f1_list[i-1], f1_list[i], f1_list[i+1], 
                                            xyz1_list[i-1], xyz1_list[i], xyz1_list[i+1],
                                            batch1)
                f_back_2, f_forward_2 = self.local_gat2(f2_list[i-1], f2_list[i], f2_list[i+1],
                                            xyz2_list[i-1], xyz2_list[i], xyz2_list[i+1],
                                            batch2)
                f_back_3, f_forward_3 = self.local_gat3(f3_list[i-1], f3_list[i], f3_list[i+1],
                                            xyz3_list[i-1], xyz3_list[i], xyz3_list[i+1],
                                            batch3)
                H1_1, C1_1 = self.lstm_1(H1_1, C1_1, f_back_1, f_forward_1, f1_list[i])
                H1_2, C1_2 = self.lstm_2(H1_2, C1_2, f_back_2, f_forward_2, f2_list[i])
                H1_3, C1_3 = self.lstm_3(H1_3, C1_3, f_back_3, f_forward_3, f3_list[i])

                prev_H1_1, prev_H1_2, prev_H1_3 = H1_1, H1_2, H1_3 # H after[f34]
                prev_C1_1, prev_C1_2, prev_C1_3 = C1_1, C1_2, C1_3 # C after[f34]

            else:
                # [f4, f44, f43]
                f_back_1, f_forward_1 = self.local_gat1(f1_list[i-1], f1_list[i], f1_list[i], 
                                            xyz1_list[i-1], xyz1_list[i], xyz1_list[i],
                                            batch1)
                f_back_2, f_forward_2 = self.local_gat2(f2_list[i-1], f2_list[i], f2_list[i],
                                            xyz2_list[i-1], xyz2_list[i], xyz2_list[i],
                                            batch2)
                f_back_3, f_forward_3 = self.local_gat3(f3_list[i-1], f3_list[i], f3_list[i],
                                            xyz3_list[i-1], xyz3_list[i], xyz3_list[i],
                                            batch3)
                # H and C after f44                       
                H1_1, C1_1 = self.lstm_1(H1_1, C1_1, f_back_1, f_forward_1, f1_list[i])
                H1_2, C1_2 = self.lstm_2(H1_2, C1_2, f_back_2, f_forward_2, f2_list[i])
                H1_3, C1_3 = self.lstm_3(H1_3, C1_3, f_back_3, f_forward_3, f3_list[i])

        last_in_xyz = (input_xyz_list[-1].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
        last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
        # inference
        x2, pos_2, batch_2 = self.fp32(H1_3, last_xyz3, batch3, H1_2, last_xyz2, batch2)
        x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, H1_1, last_xyz1, batch1)
        x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)
        
        pc_next = self.refiner(x=x0, pos=pos_0, batch=batch_0)
        # pc_next = last_in_xyz + self.classifier4(self.classifier3(self.classifier2(self.classifier1(x0.T)))).T # p5_pred
        # print("pc_next.shape",pc_next.shape)
        pred_detail_xyz_list.append(pc_next)


        for i in range(1, num_pred):
            fea1, xyz1, batch1 = self.sa1(pc_next, pc_next, in_batch)
            f1_list.append(fea1)
            xyz1_list.append(xyz1)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            f2_list.append(fea2)
            xyz2_list.append(xyz2)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
            f3_list.append(fea3)
            xyz3_list.append(xyz3)
            # [0, ... , 3, 4, 5_pred]

            # compute for [f4, f45, f43]
            f_back_1, f_forward_1 = self.local_gat1(f1_list[-3], f1_list[-2], f1_list[-1], 
                                            xyz1_list[-3], xyz1_list[-2], xyz1_list[-1],
                                            batch1)
            f_back_2, f_forward_2 = self.local_gat2(f2_list[-3], f2_list[-2], f2_list[-1],
                                        xyz2_list[-3], xyz2_list[-2], xyz2_list[-1],
                                        batch2)
            f_back_3, f_forward_3 = self.local_gat3(f3_list[-3], f3_list[-2], f3_list[-1],
                                        xyz3_list[-3], xyz3_list[-2], xyz3_list[-1],
                                        batch3)
            #input of lstm: [f45, f43, f4], pos=5
            H1_1, C1_1 = self.lstm_1(prev_H1_1, prev_C1_1, f_back_1, f_forward_1, f1_list[-2])
            H1_2, C1_2 = self.lstm_2(prev_H1_2, prev_C1_2, f_back_2, f_forward_2, f2_list[-2])
            H1_3, C1_3 = self.lstm_3(prev_H1_3, prev_C1_3, f_back_3, f_forward_3, f3_list[-2])

            # store H and C after [f45, f43, f4, pos=5]
            prev_H1_1, prev_H1_2, prev_H1_3 = H1_1, H1_2, H1_3
            prev_C1_1, prev_C1_2, prev_C1_3 = C1_1, C1_2, C1_3

            # compute for [f55, f54]
            f_back_1, f_forward_1 = self.local_gat1(f1_list[-2], f1_list[-1], f1_list[-1], 
                                            xyz1_list[-2], xyz1_list[-1], xyz1_list[-1],
                                            batch1)
            f_back_2, f_forward_2 = self.local_gat2(f2_list[-2], f2_list[-1], f2_list[-1],
                                        xyz2_list[-2], xyz2_list[-1], xyz2_list[-1],
                                        batch2)
            f_back_3, f_forward_3 = self.local_gat3(f3_list[-2], f3_list[-1], f3_list[-1],
                                        xyz3_list[-2], xyz3_list[-1], xyz3_list[-1],
                                        batch3)

            # input of lstm: [f55, f54, f5, pos=6]
            H1_1, C1_1 = self.lstm_1(H1_1, C1_1, f_back_1, f_forward_1, f1_list[-1])
            H1_2, C1_2 = self.lstm_2(H1_2, C1_2, f_back_2, f_forward_2, f2_list[-1])
            H1_3, C1_3 = self.lstm_3(H1_3, C1_3, f_back_3, f_forward_3, f3_list[-1])

            last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
            x2, pos_2, batch_2 = self.fp32(H1_3, last_xyz3, batch3, H1_2, last_xyz2, batch2)
            x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, H1_1, last_xyz1, batch1)
            x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, pc_next, in_batch)

            pc_next = self.refiner(x=x0, pos=pos_0, batch=batch_0)
            # print("pc_next.shape",pc_next.shape)
            pred_detail_xyz_list.append(pc_next)
    
        return pred_detail_xyz_list
    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device",device)
    data_config = {'root': "/usr/stud/ype/storage/user/kittidata/dataset/sequences", 'npoints': 16384, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00']}
    demo_dataset = KittiDataset_2(root=data_config['root'], npoints=16384, input_num=5, pred_num=5, seqs=['00'])
    train_dataloader = DataLoader(demo_dataset, batch_size=1) 
    bin_xyz, _ = next(iter(train_dataloader))
    bin_xyz = [i.to(device) for i in bin_xyz]


    #  input_xyz_list, num_pred, device

    testmodel = PC_MO_NOT_R().to(device)
    res = testmodel(bin_xyz, 5, device)
    for i in res:
        print("i.shape",i.shape)
        










        




            








        
