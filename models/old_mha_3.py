#old mha, 3行都有
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


class MHA_l3(nn.Module):
    def __init__(self, use_knn=True):
        super(MHA_l3, self).__init__()
        self.sa1 = SAModule(float(1/32), MLP([3+3, 32, 32, 64],norm=None), r=0.2*8.5, k=16, knn=use_knn)
        self.sa2 = SAModule(0.5, MLP([64+3, 96, 96, 128],norm=None), r=0.4*8.5, k=16, knn=use_knn)
        self.sa3 = SAModule(0.5, MLP([128+3, 128, 128, 256],norm=None), r=0.8*8.5, k=8, knn=use_knn)

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.num_heads = 4

        self.local_gat1 = LPT_BD_not(32, self.C1 , self.C1)
        self.local_gat2 = LPT_BD_not(16, self.C2 , self.C2)
        self.local_gat3 = LPT_BD_not(8, self.C3 , self.C3)


        self.seq_att1_b = nn.MultiheadAttention(self.C1, self.num_heads)
        self.seq_att2_b = nn.MultiheadAttention(self.C2, self.num_heads)
        self.seq_att3_b = nn.MultiheadAttention(self.C3, self.num_heads)

        self.seq_att1_f = nn.MultiheadAttention(self.C1, self.num_heads)
        self.seq_att2_f = nn.MultiheadAttention(self.C2, self.num_heads)
        self.seq_att3_f = nn.MultiheadAttention(self.C3, self.num_heads)

        self.seq_att1_c = nn.MultiheadAttention(self.C1, self.num_heads)
        self.seq_att2_c = nn.MultiheadAttention(self.C2, self.num_heads)
        self.seq_att3_c = nn.MultiheadAttention(self.C3, self.num_heads)

        self.proj_1 = MLP([3*self.C1, 2*self.C1, self.C1, self.C1], norm=None)
        self.proj_2 = MLP([3*self.C2, 2*self.C2, self.C2, self.C2], norm=None)
        self.proj_3 = MLP([3*self.C3, 2*self.C3, self.C3, self.C3], norm=None)

        self.seq_cur_1 = MLP([2*self.C1, self.C1, self.C1], norm=None)
        self.seq_cur_2 = MLP([2*self.C2, self.C2, self.C2], norm=None)
        self.seq_cur_3 = MLP([2*self.C3, self.C3, self.C3], norm=None)


        self.fp32 = FPModule(8, MLP([(256+128), 256*2, 256, 256], norm=None))
        self.fp21 = FPModule(16, MLP([256+64, 256, 256],norm=None))
        self.fp10 = FPModule(16, MLP([256, 256, 128], norm=None))

      
        self.classifier1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1, bias=False)
        self.classifier4 = nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1, bias=False)

    
    def forward(self, input_xyz_list, num_pred, device):
        
        pred_detail_xyz_list = []

        batch_size, df, N = input_xyz_list[0].shape
        t = torch.LongTensor(range(batch_size)).to(device)
        in_batch = torch.repeat_interleave(t, N).to(device)

        xyz1_list = []
        xyz2_list = []
        xyz3_list = []


        ## encoder
        for i in range(len(input_xyz_list)):
            if i ==0:
                xyz0 = (input_xyz_list[i].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
                fea1, xyz1, batch1 = self.sa1(xyz0, xyz0, in_batch)
                fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
                fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)

                n1, h1 = fea1.shape
                n2, h2 = fea2.shape
                n3, h3 = fea3.shape

                f1_list = torch.empty((len(input_xyz_list), n1, h1)).to(device)
                f2_list = torch.empty((len(input_xyz_list), n2, h2)).to(device)
                f3_list = torch.empty((len(input_xyz_list), n3, h3)).to(device)

                f1_list[i] = fea1
                f2_list[i] = fea2
                f3_list[i] = fea3

                xyz1_list.append(xyz1)
                xyz2_list.append(xyz2)
                xyz3_list.append(xyz3)

            else:

                xyz0 = (input_xyz_list[i].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
                fea1, xyz1, batch1 = self.sa1(xyz0, xyz0, in_batch)
                f1_list[i] = fea1
                xyz1_list.append(xyz1)
                fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
                f2_list[i] = fea2
                xyz2_list.append(xyz2)
                fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
                f3_list[i] = fea3
                xyz3_list.append(xyz3)


        # [f01, f00]
        f_back_1, f_forward_1  = self.local_gat1(f1_list[0], f1_list[0], f1_list[1], 
                                        xyz1_list[0], xyz1_list[0], xyz1_list[1],
                                        batch1)
        f_back_2, f_forward_2 = self.local_gat2(f2_list[0], f2_list[0], f2_list[1],
                                        xyz2_list[0], xyz2_list[0], xyz2_list[1],
                                        batch2)
        f_back_3, f_forward_3 = self.local_gat3(f3_list[0], f3_list[0], f3_list[1],
                                        xyz3_list[0], xyz3_list[0], xyz3_list[1],
                                        batch3)
        n1, h1 = f_back_1.shape
        n2, h2 = f_back_2.shape
        n3, h3 = f_back_3.shape

        f_back_1_list = torch.empty((len(f1_list)-1, n1, h1)).to(device)
        f_forward_1_list = torch.empty((len(f1_list)-1, n1, h1)).to(device)
        f_back_2_list = torch.empty((len(f2_list)-1, n2, h2)).to(device)
        f_forward_2_list = torch.empty((len(f2_list)-1, n2, h2)).to(device)
        f_back_3_list = torch.empty((len(f3_list)-1, n3, h3)).to(device)
        f_forward_3_list = torch.empty((len(f3_list)-1, n3, h3)).to(device)

        f_back_1_list[0] = f_back_1
        f_forward_1_list[0] = f_forward_1
        f_back_2_list[0] = f_back_2
        f_forward_2_list[0] = f_forward_2
        f_back_3_list[0] = f_back_3
        f_forward_3_list[0] = f_forward_3

        for i in range(1, len(f1_list)-1):
            f_back_1, f_forward_1 = self.local_gat1(f1_list[i-1], f1_list[i], f1_list[i+1], 
                                            xyz1_list[i-1], xyz1_list[i], xyz1_list[i+1],
                                            batch1)
            f_back_2, f_forward_2 = self.local_gat2(f2_list[i-1], f2_list[i], f2_list[i+1],
                                        xyz2_list[i-1], xyz2_list[i], xyz2_list[i+1],
                                        batch2)
            f_back_3, f_forward_3 = self.local_gat3(f3_list[i-1], f3_list[i], f3_list[i+1],
                                        xyz3_list[i-1], xyz3_list[i], xyz3_list[i+1],
                                        batch3)
            f_back_1_list[i] = f_back_1
            f_forward_1_list[i] = f_forward_1
            f_back_2_list[i] = f_back_2
            f_forward_2_list[i] = f_forward_2
            f_back_3_list[i] = f_back_3
            f_forward_3_list[i] = f_forward_3

        q_back_1, q_forward_1 = self.local_gat1(f1_list[-2], f1_list[-1], f1_list[-1], 
                                            xyz1_list[-2], xyz1_list[-1], xyz1_list[-1],
                                            batch1)
        q_back_2, q_forward_2 = self.local_gat2(f2_list[-2], f2_list[-1], f2_list[-1],
                                        xyz2_list[-2], xyz2_list[-1], xyz2_list[-1],
                                        batch2)
        q_back_3, q_forward_3 = self.local_gat3(f3_list[-2], f3_list[-1], f3_list[-1],
                                        xyz3_list[-2], xyz3_list[-1], xyz3_list[-1],
                                        batch3)
        

        tm1_b, _ = self.seq_att1_b(q_back_1.unsqueeze(0), f_back_1_list, f_back_1_list)
        tm2_b, _ = self.seq_att2_b(q_back_2.unsqueeze(0), f_back_2_list, f_back_2_list)
        tm3_b, _ = self.seq_att3_b(q_back_3.unsqueeze(0), f_back_3_list, f_back_3_list)

        tm1_f, _ = self.seq_att1_f(q_forward_1.unsqueeze(0), f_forward_1_list, f_forward_1_list)
        tm2_f, _ = self.seq_att2_f(q_forward_2.unsqueeze(0), f_forward_2_list, f_forward_2_list)
        tm3_f, _ = self.seq_att3_f(q_forward_3.unsqueeze(0), f_forward_3_list, f_forward_3_list)

        tm1_c, _ = self.seq_att1_c(f1_list[-1].unsqueeze(0), f1_list[:-1], f1_list[:-1])
        tm2_c, _ = self.seq_att2_c(f2_list[-1].unsqueeze(0), f2_list[:-1], f2_list[:-1])
        tm3_c, _ = self.seq_att3_c(f3_list[-1].unsqueeze(0), f3_list[:-1], f3_list[:-1])

        tm1_b = tm1_b.squeeze(0)
        tm2_b = tm2_b.squeeze(0)
        tm3_b = tm3_b.squeeze(0)

        tm1_f = tm1_f.squeeze(0)
        tm2_f = tm2_f.squeeze(0)
        tm3_f = tm3_f.squeeze(0)

        tm1_c = tm1_c.squeeze(0)
        tm2_c = tm2_c.squeeze(0)
        tm3_c = tm3_c.squeeze(0)



        tm1 = self.proj_1(torch.cat((tm1_b, tm1_c, tm1_f), dim=-1))
        tm2 = self.proj_2(torch.cat((tm2_b, tm2_c, tm2_f), dim=-1))
        tm3 = self.proj_3(torch.cat((tm3_b, tm3_c, tm3_f), dim=-1))
        
        out1 = self.seq_cur_1(torch.cat((tm1, f1_list[-1]), dim=-1))
        out2 = self.seq_cur_2(torch.cat((tm2, f2_list[-1]), dim=-1))
        out3 = self.seq_cur_3(torch.cat((tm3, f3_list[-1]), dim=-1))


        
        last_in_xyz = (input_xyz_list[-1].permute(0, 2, 1).contiguous()).view(-1,3).contiguous()
        last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
        
        x2, pos_2, batch_2 = self.fp32(out3, last_xyz3, batch3, out2, last_xyz2, batch2)
        x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, out1, last_xyz1, batch1)
        x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)
        pc_next = last_in_xyz + self.classifier4(self.classifier3(self.classifier2(self.classifier1(x0.T)))).T
        # print("pc_next.shape",pc_next.shape)
        pred_detail_xyz_list.append(pc_next)


        for i in range(num_pred-1):
            last_in_xyz = pred_detail_xyz_list[-1]

            fea1, xyz1, batch1 = self.sa1(last_in_xyz, pc_next, in_batch)
            fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
            fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)

            f1_list = torch.cat([f1_list, fea1.unsqueeze(0)], dim=0)
            f2_list = torch.cat([f2_list, fea2.unsqueeze(0)], dim=0)
            f3_list = torch.cat([f3_list, fea3.unsqueeze(0)], dim=0)

            xyz1_list.append(xyz1)
            xyz2_list.append(xyz2)
            xyz3_list.append(xyz3)
            
            f_back_1, f_forward_1 = self.local_gat1(f1_list[-3], f1_list[-2], f1_list[-1], 
                                            xyz1_list[-3], xyz1_list[-2], xyz1_list[-1],
                                            batch1)
            f_back_2, f_forward_2 = self.local_gat2(f2_list[-3], f2_list[-2], f2_list[-1],
                                        xyz2_list[-3], xyz2_list[-2], xyz2_list[-1],
                                        batch2)
            f_back_3, f_forward_3 = self.local_gat3(f3_list[-3], f3_list[-2], f3_list[-1],
                                        xyz3_list[-3], xyz3_list[-2], xyz3_list[-1],
                                        batch3)
            
            f_back_1_list = torch.cat([f_back_1_list, f_back_1.unsqueeze(0)], dim=0)
            f_forward_1_list = torch.cat([f_forward_1_list, f_forward_1.unsqueeze(0)], dim=0)
            f_back_2_list = torch.cat([f_back_2_list, f_back_2.unsqueeze(0)], dim=0)
            f_forward_2_list = torch.cat([f_forward_2_list, f_forward_2.unsqueeze(0)], dim=0)
            f_back_3_list = torch.cat([f_back_3_list, f_back_3.unsqueeze(0)], dim=0)
            f_forward_3_list = torch.cat([f_forward_3_list, f_forward_3.unsqueeze(0)], dim=0)


            q_back_1, q_forward_1 = self.local_gat1(f1_list[-2], f1_list[-1], f1_list[-1], 
                                        xyz1_list[-2], xyz1_list[-1], xyz1_list[-1],
                                        batch1)
            q_back_2, q_forward_2 = self.local_gat2(f2_list[-2], f2_list[-1], f2_list[-1],
                                            xyz2_list[-2], xyz2_list[-1], xyz2_list[-1],
                                            batch2)
            q_back_3, q_forward_3 = self.local_gat3(f3_list[-2], f3_list[-1], f3_list[-1],
                                            xyz3_list[-2], xyz3_list[-1], xyz3_list[-1],
                                            batch3)
            

            tm1_b, _ = self.seq_att1_b(q_back_1.unsqueeze(0), f_back_1_list, f_back_1_list)
            tm2_b, _ = self.seq_att2_b(q_back_2.unsqueeze(0), f_back_2_list, f_back_2_list)
            tm3_b, _ = self.seq_att3_b(q_back_3.unsqueeze(0), f_back_3_list, f_back_3_list)

            tm1_f, _ = self.seq_att1_f(q_forward_1.unsqueeze(0), f_forward_1_list, f_forward_1_list)
            tm2_f, _ = self.seq_att2_f(q_forward_2.unsqueeze(0), f_forward_2_list, f_forward_2_list)
            tm3_f, _ = self.seq_att3_f(q_forward_3.unsqueeze(0), f_forward_3_list, f_forward_3_list)

            tm1_c, _ = self.seq_att1_c(f1_list[-1].unsqueeze(0), f1_list[:-1], f1_list[:-1])
            tm2_c, _ = self.seq_att2_c(f2_list[-1].unsqueeze(0), f2_list[:-1], f2_list[:-1])
            tm3_c, _ = self.seq_att3_c(f3_list[-1].unsqueeze(0), f3_list[:-1], f3_list[:-1])

            tm1_b = tm1_b.squeeze(0)
            tm2_b = tm2_b.squeeze(0)
            tm3_b = tm3_b.squeeze(0)

            tm1_f = tm1_f.squeeze(0)
            tm2_f = tm2_f.squeeze(0)
            tm3_f = tm3_f.squeeze(0)

            tm1_c = tm1_c.squeeze(0)
            tm2_c = tm2_c.squeeze(0)
            tm3_c = tm3_c.squeeze(0)


            tm1 = self.proj_1(torch.cat((tm1_b, tm1_c, tm1_f), dim=-1))
            tm2 = self.proj_2(torch.cat((tm2_b, tm2_c, tm2_f), dim=-1))
            tm3 = self.proj_3(torch.cat((tm3_b, tm3_c, tm3_f), dim=-1))
            
            out1 = self.seq_cur_1(torch.cat((tm1, f1_list[-1]), dim=-1))
            out2 = self.seq_cur_2(torch.cat((tm2, f2_list[-1]), dim=-1))
            out3 = self.seq_cur_3(torch.cat((tm3, f3_list[-1]), dim=-1))


            
            last_xyz1, last_xyz2, last_xyz3 = xyz1_list[-1], xyz2_list[-1], xyz3_list[-1]
            
            x2, pos_2, batch_2 = self.fp32(out3, last_xyz3, batch3, out2, last_xyz2, batch2)
            x1, pos_1, batch_1 = self.fp21(x2, pos_2, batch_2, out1, last_xyz1, batch1)
            x0, pos_0, batch_0 = self.fp10(x1, pos_1, batch_1, None, last_in_xyz, in_batch)
            pc_next = last_in_xyz + self.classifier4(self.classifier3(self.classifier2(self.classifier1(x0.T)))).T
            # print("pc_next.shape",pc_next.shape)
            pred_detail_xyz_list.append(pc_next)
                    

        return pred_detail_xyz_list
    

        


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

    testmodel = MHA_l3(att_args).to(device)

    res = testmodel(bin_xyz, 5, device)
    for i in res:
        print("i.shape",i.shape)
        










        




            








        
