import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph, knn

from torch_geometric.nn import global_max_pool, global_mean_pool, GATv2Conv
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


from pugcn_lib.upsample import GeneralUpsampler
from pugcn_lib.models import Refiner



from models.utils import  PC_Encoder, LPT_raw
    
class PC_forecasting_model_knn_not(nn.Module):
    def __init__(self, encoder_args, att_args, upsampler_args, refiner_args, npoint):

        super(PC_forecasting_model_knn_not, self).__init__()
        self.encoder = PC_Encoder()

        self.local_fea_dim = 256

        self.num_heads = att_args['num_heads']
        self.num_neighs = att_args['num_neighs']

        self.local_agg_model = LPT_raw(self.num_neighs, 256, 256)
        
        self.mul_fr_att = nn.MultiheadAttention(256, self.num_heads)

        self.coarse_re = nn.Sequential(
            nn.Linear(256,64),
            nn.LeakyReLU(),
            nn.Linear(64,3)
        )

        self.detail_re = nn.Sequential(
            nn.Linear(upsampler_args['out_channels'],int(upsampler_args['out_channels']/4)),
            nn.LeakyReLU(),
            nn.Linear(int(upsampler_args['out_channels']/4),int(upsampler_args['out_channels']/16)),
            nn.LeakyReLU(),
            nn.Linear(int(upsampler_args['out_channels']/16),3)
        )

        self.upsampler = GeneralUpsampler(**upsampler_args)
        self.refiner = Refiner(**refiner_args)


    def forward(self, input_xyz_list, num_pred, device):

        #input_xyz_list: [(5, 3, N)]
        #input_fea_list: [(5, d_f, N)]

        pred_coares_xyz_list = []
        pred_detail_xyz_list = []

        batch_size, df, N = input_xyz_list[0].shape
        t = torch.LongTensor(range(batch_size)).to(device)
        in_batch = torch.repeat_interleave(t, N).to(device) # (N')

        len_input = len(input_xyz_list)
        encoded_xyz_list = []
        encoded_fea_list = []
        for i in range(len_input):
            input_xyz = (input_xyz_list[i].permute(0,2,1).contiguous()).view(-1,3).contiguous()
            # input_fea = input_fea_list[i] #(B, d_f, N)
            en_fea, en_xyz, en_batch = self.encoder(input_xyz, input_xyz, in_batch)
            
            encoded_xyz_list.append(en_xyz)
            encoded_fea_list.append(en_fea)
        
        last_fea = encoded_fea_list[-1]
        last_xyz = encoded_xyz_list[-1]

        n, h = last_fea.shape
        
        f_ij_list = torch.empty((len(encoded_fea_list), n, h)).to(device)

        for i in range(len(encoded_fea_list)):
                f_ij_list[i,:,:] = self.local_agg_model(encoded_fea_list[i], last_fea, 
                                                        encoded_xyz_list[i], last_xyz, 
                                                        en_batch)

        fr_q = f_ij_list[-1].unsqueeze(0) #(1, B*N', 256)
        # print("fr_q.shape", fr_q.shape)
        fr_k = f_ij_list
        fr_v = f_ij_list

        fr_out, _ = self.mul_fr_att(fr_q, fr_k, fr_v) #(1, B*N', 256)
        # print("fr_out.shape", fr_out.shape)
        fea_next =  fr_out.squeeze(0)   #(B*N', 256) 


        next_coarse_xyz = self.coarse_re(fea_next)+last_xyz #(B*N', 3)

        # print("next_coarse_xyz .shape", next_coarse_xyz.shape)
        # coarse_xyz = coarse_xyz.reshape(batch_size, 1024, 3).permute(0,2,1) #(B, 3, N')
        pred_coares_xyz_list.append(next_coarse_xyz)

        fea_upsampled , up_batch = self.upsampler(x=fea_next, pos = next_coarse_xyz, batch=en_batch, return_batch=True)

        xyz_upsampled = self.detail_re(fea_upsampled) #(B*N, 3)

        next_detail_xyz = self.refiner(x = fea_upsampled, pos = xyz_upsampled, batch=up_batch) #(B*N, 3)
        # print("next_detail_xyz.shape", next_detail_xyz.shape)
        # next_detail_xyz = next_detail_xyz.reshape(batch_size, 1024, 3).permute(0,2,1) #(B, 3, N')
        pred_detail_xyz_list.append(next_detail_xyz)


        for _ in range(num_pred-1):
            input_xyz = pred_detail_xyz_list[-1]
            en_fea, en_xyz, en_batch = self.encoder(input_xyz, input_xyz, in_batch)
            encoded_xyz_list.append(en_xyz)
            encoded_fea_list.append(en_fea)

            last_fea = encoded_fea_list[-1]
            last_xyz = encoded_xyz_list[-1]

            n, h = last_fea.shape
            
            f_ij_list = torch.empty((len(encoded_fea_list), n, h)).to(device)

            for i in range(len(encoded_fea_list)):
                f_ij_list[i,:,:] = self.local_agg_model(encoded_fea_list[i], last_fea, 
                                                        encoded_xyz_list[i], last_xyz, 
                                                        en_batch)

            fr_q = f_ij_list[-1].unsqueeze(0) #(1, B*N', 256)
            fr_k = f_ij_list
            fr_v = f_ij_list

            fr_out, _ = self.mul_fr_att(fr_q, fr_k, fr_v) #(1, B*N', 256)
            fea_next =  fr_out.squeeze(0)   #(B*N', 256)


            next_coarse_xyz = self.coarse_re(fea_next)+last_xyz #(B*N', 3)
            # print("next_coarse_xyz .shape", next_coarse_xyz.shape)
            # coarse_xyz = coarse_xyz.reshape(batch_size, npoint, 3).permute(0,2,1) #(B, 3, N')
            pred_coares_xyz_list.append(next_coarse_xyz)

            fea_upsampled , up_batch = self.upsampler(x=fea_next, pos = next_coarse_xyz, batch=en_batch, return_batch=True)

            xyz_upsampled = self.detail_re(fea_upsampled) #(B*N, 3)

            next_detail_xyz = self.refiner(x = fea_upsampled, pos = xyz_upsampled, batch=up_batch) #(B*N, 3)
            # print("next_detail_xyz.shape", next_detail_xyz.shape)
            # next_detail_xyz = next_detail_xyz.reshape(batch_size, npoint, 3).permute(0,2,1) #(B, 3, N')
            pred_detail_xyz_list.append(next_detail_xyz)

        return pred_coares_xyz_list, pred_detail_xyz_list




if __name__ == "__main__":
    pass
    
    








    







