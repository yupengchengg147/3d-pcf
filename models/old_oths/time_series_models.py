import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_cluster import knn_graph, knn
from torch import Tensor
from typing import Optional, Tuple
from torch_geometric.nn import global_max_pool, global_mean_pool, GATv2Conv
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from torch.utils.data import DataLoader

from dataset.kitti_dataset import KittiDataset

from pugcn_lib.feature_extractor import InceptionFeatureExtractor
from pugcn_lib.upsample import GeneralUpsampler
from pugcn_lib.models import Refiner



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn

class Agg_btw_frames(nn.Module):
    def __init__(self,dim,num_heads, num_neighs):
        super(Agg_btw_frames, self).__init__()
        self.dim = dim
        self.m_att = MultiHeadAttention(dim, num_heads)
        self.num_neighs = num_neighs
    
    def forward(self, fea_i, fea_j, batch, pos_i=None, pos_j=None):
        """
        pos_i, j: (B*N, 3)
        fea_i, fea_j: (B*N, dim)
        batch: LongTensor(B*N)
        eg: t = torch.LongTensor([0,1])
            dbatch = torch.repeat_interleave(t, 1024)
        """
        if (pos_i is not None) and (pos_j is not None):
            y_to_x_index = knn(pos_i, pos_j, self.num_neighs,batch, batch)
        else:
            y_to_x_index = knn(fea_i, fea_j, self.num_neighs,batch, batch)

        out = torch.empty_like(fea_j)
        
        for i in range(fea_j.shape[-2]):
            f1_index_i = y_to_x_index[1][y_to_x_index[0,:]==i] # f5_i 根据 pos 找到的 10个f1中的neighbors
            q_i = fea_j[i,:][None, None,:]
            k_i = fea_i[f1_index_i,:][None, :,:]
            v_i = fea_i[f1_index_i,:][None, :,:]
            context_i,_ = self.m_att(q_i, k_i, v_i)
            # print(context.shape)
            out[i,:] = context_i.view(-1, self.dim)

        return(out)

class PC_forecasting_model_0_0(nn.Module):
    def __init__(self, encoder_args, encoder_kargs, dim_args, att_args, upsampler_args, refiner_args):

        super(PC_forecasting_model_0_0, self).__init__()
        self.encoder = InceptionFeatureExtractor(channels= encoder_args['channels'], k= encoder_args['k'],
                                                 dilations=encoder_args['dilations'],
                                                 n_idgcn_blocks= encoder_args['n_idgcn_blocks'],
                                                 n_dgcn_blocks= encoder_args['n_dgcn_blocks'],
                                                 radio= encoder_args['radio'],
                                                 **encoder_kargs
                                                 )

        self.global_fea_dim = dim_args['global_fea_dim']
        self.local_fea_dim = encoder_args['channels']

        self.encoder_ratio = encoder_args['radio']
        self.num_heads = att_args['num_heads']
        self.num_neighs = att_args['num_neighs']

        # self.knn_with_pos = att_args['pos']
        
        self.node_level_proj = nn.Sequential(nn.Linear(self.local_fea_dim, 128),nn.LeakyReLU(), nn.Linear(128, self.global_fea_dim), nn.LeakyReLU())
        
        self.fr_level_att = ScaledDotProductAttention(self.global_fea_dim)
        # self.local_agg_model = Agg_btw_frames(dim=self.local_fea_dim, num_heads=self.num_heads, num_neighs=self.num_neighs)
        self.local_agg_model = GATv2Conv(in_channels=(self.local_fea_dim, self.local_fea_dim), out_channels=self.local_fea_dim, heads=self.num_heads, concat=False, bias=False, add_self_loops=False, flow='target_to_source')

        self.upsampler = GeneralUpsampler(k=upsampler_args['k'], r=upsampler_args['r'], in_channels=upsampler_args['in_channels'], out_channels=upsampler_args['out_channels'], upsampler=upsampler_args['upsampler'])
        self.reconstructor = nn.Sequential(torch.nn.Linear(self.local_fea_dim, self.local_fea_dim), torch.nn.LeakyReLU(), torch.nn.Linear(self.local_fea_dim, 3), torch.nn.LeakyReLU())

        self.refiner = Refiner(in_channels=refiner_args['in_channels'], out_channels=refiner_args['out_channels'], k=refiner_args['k'], dilations=refiner_args['dilations'], add_points=refiner_args['add_points'])



    def forward(self, input_pc_seq, batch_size, num_pred, num_points, device):

        # print("input_pc_seq.shape: ", input_pc_seq.shape)
        # print("should be equal to: ", batch_size*num_points)
        
        assert input_pc_seq.shape[1] == int(batch_size*num_points), "input_pc_seq shape is not correct"

        len_input = input_pc_seq.shape[0]

        t = torch.LongTensor(range(batch_size)).to(device)
        in_batch = torch.repeat_interleave(t, num_points).to(device) # (2, num_points)
        encoder_batch = torch.repeat_interleave(t, int(num_points * self.encoder_ratio)).to(device) # (2, num_points * encoder_ratio)
    
        pc_encoded_seq = torch.empty(len_input, int(num_points*batch_size*self.encoder_ratio), self.local_fea_dim).to(device) #(5, B*N, local_fea_dim)
        global_fea_seq = torch.empty(len_input, batch_size, self.global_fea_dim).to(device) #(5, B, global_fea_dim)

        for i in range(len_input):
            pc = input_pc_seq[i,:,:] #(B*N, 3)
            if i == len(input_pc_seq)-1:
                pc_encoded, last_frame_idx = self.encoder(x=pc, pos = pc, batch=in_batch, return_index = True) #(B*N, local_fea_dim)
                last_frame_idx = last_frame_idx.long()

                assert last_frame_idx.max() <= pc.shape[0], "last_frame_idx is not correct"
                assert pc_encoded.shape[0] == int(num_points * batch_size * self.encoder_ratio), "pc_encoded shape is not correct"

            else:   
                pc_encoded = self.encoder(x=pc, pos = pc, batch=in_batch, return_index = False) #(B*N, local_fea_dim)
                assert pc_encoded.shape[0] == int(num_points * batch_size * self.encoder_ratio), "pc_encoded shape is not correct"

            pc_encoded_seq[i,:,:] = pc_encoded

            pc_pooled = global_mean_pool(self.node_level_proj(pc_encoded), batch=encoder_batch) #(B, global_fea_dim)
            assert pc_pooled.shape == torch.Size([batch_size, self.global_fea_dim]), "pc_pooled shape is not correct"
            global_fea_seq[i,:,:] = pc_pooled


        q = global_fea_seq[-1,:,:].unsqueeze(1) # (B, 1, C)
        k = global_fea_seq[0:-1,:,:].permute(1,0,2) # (B, T-1, C)
        v = global_fea_seq[0:-1,:,:].permute(1,0,2)  # (B, T-1, C)
        
        _, frame_level_att = self.fr_level_att(q,k,v) # 再次利用frame_level_aggregated_global_fea???

        res = torch.empty(batch_size,len(pc_encoded_seq)-1, int(num_points * self.encoder_ratio), self.local_fea_dim).to(device) #$ (B, T-1, N, local_fea_dim)

        for i in range(len(pc_encoded_seq)-1):

            by_to_x_index = knn(x=pc_encoded_seq[i,:,:],y=pc_encoded_seq[-1,:,:], k=self.num_neighs, batch_x=encoder_batch, batch_y=encoder_batch)
            assert by_to_x_index[1].max() <= pc_encoded_seq[i,:,:].shape[0] , "by_to_x_index shape is not correct"
            out = self.local_agg_model((pc_encoded_seq[-1,:,:],pc_encoded_seq[i,:,:]),by_to_x_index)

            # out = self.local_agg_model(pc_encoded_seq[i,:,:], pc_encoded_seq[-1,:,:], encoder_batch) # (B*N, local_fea_dim)
            assert out.shape == torch.Size([int(num_points * batch_size * self.encoder_ratio), self.local_fea_dim]), "out shape is not correct"
            res[:,i,:,:] = out.view(batch_size,int(num_points * self.encoder_ratio), self.local_fea_dim)

        fr_weight = frame_level_att.permute(0,2,1).unsqueeze(-1)
        assert fr_weight.shape == torch.Size([batch_size, len(pc_encoded_seq)-1, 1, 1]), "fr_weight shape is not correct"
        weighted_res = torch.sum(torch.mul(res, fr_weight),dim=1)
        assert weighted_res.shape == torch.Size([batch_size, int(num_points * self.encoder_ratio), self.local_fea_dim]), "weighted_res shape is not correct"

        tm_out = torch.cat([pc_encoded_seq[-1], weighted_res.view(-1,self.local_fea_dim)], dim=-1) # (B*N/r, local_fea_dim*2)
        assert tm_out.shape == torch.Size([int(num_points * batch_size * self.encoder_ratio), self.local_fea_dim*2]), "tm_out shape is not correct"

        upsampled_pc, up_batch = self.upsampler(x=tm_out, pos = input_pc_seq[-1,last_frame_idx,:],batch=encoder_batch, return_batch=True)
        pc_reconstructed = self.reconstructor(upsampled_pc)
        pc_f1 = self.refiner(x = upsampled_pc, pos = pc_reconstructed, batch=up_batch)

        assert pc_f1.shape == torch.Size([int(num_points * batch_size), 3]), "pc_f1 shape is not correct"

        pred_pc_seq = []
        pred_pc_seq.append(pc_f1)

        
        for _ in range(num_pred-1):
            
            pc_encoded, last_frame_idx = self.encoder(x=pred_pc_seq[-1], pos = pred_pc_seq[-1], batch=in_batch, return_index = True) #(B*N, local_fea_dim) （b*n,dim）
            last_frame_idx = last_frame_idx.long()
            assert last_frame_idx.max() <= pred_pc_seq[-1].shape[0], "last_frame_idx is not correct"
            assert pc_encoded.shape[0] == int(num_points * batch_size * self.encoder_ratio), "pc_encoded shape is not correct"

            pc_encoded_seq = torch.cat([pc_encoded_seq, pc_encoded.unsqueeze(0)], dim=0)

            pc_pooled = global_mean_pool(self.node_level_proj(pc_encoded), batch=encoder_batch) #(B,global_fea_dim)
            assert pc_pooled.shape == torch.Size([batch_size, self.global_fea_dim]), "pc_pooled shape is not correct"

            global_fea_seq = torch.cat([global_fea_seq, pc_pooled.unsqueeze(0)], dim=0)

            q = global_fea_seq[-1,:,:].unsqueeze(1) # (B, 1, C)
            k = global_fea_seq[0:-1,:,:].permute(1,0,2) # (B, T-1, C)
            v = global_fea_seq[0:-1,:,:].permute(1,0,2)  # (B, T-1, C)
            _, frame_level_att = self.fr_level_att(q,k,v) # 再次利用frame_level_aggregated_global_fea?

            res = torch.empty(batch_size,len(pc_encoded_seq)-1, int(num_points * self.encoder_ratio), self.local_fea_dim).to(device) #$ (B, T-1, N, local_fea_dim)

            for i in range(len(pc_encoded_seq)-1):
                by_to_x_index = knn(x=pc_encoded_seq[i,:,:],y=pc_encoded_seq[-1,:,:], k=self.num_neighs, batch_x=encoder_batch, batch_y=encoder_batch)
                out = self.local_agg_model((pc_encoded_seq[-1,:,:],pc_encoded_seq[i,:,:]),by_to_x_index)
                # out = self.local_agg_model(pc_encoded_seq[i,:,:], pc_encoded_seq[-1,:,:], encoder_batch) # (B*N, local_fea_dim)
                assert out.shape == torch.Size([int(num_points * batch_size * self.encoder_ratio), self.local_fea_dim]), "out shape is not correct"
                res[:,i,:,:] = out.view(batch_size,int(num_points * self.encoder_ratio), self.local_fea_dim)
            
            fr_weight = frame_level_att.permute(0,2,1).unsqueeze(-1)
            assert fr_weight.shape == torch.Size([batch_size, len(pc_encoded_seq)-1, 1, 1]), "fr_weight shape is not correct"
            weighted_res = torch.sum(torch.mul(res, fr_weight),dim=1)
            assert weighted_res.shape == torch.Size([batch_size, int(num_points * self.encoder_ratio), self.local_fea_dim]), "weighted_res shape is not correct"

            tm_out = torch.cat([pc_encoded_seq[-1], weighted_res.view(-1,self.local_fea_dim)], dim=-1) # (B*N, local_fea_dim*2)
            assert tm_out.shape == torch.Size([int(num_points * batch_size * self.encoder_ratio), self.local_fea_dim*2]), "tm_out shape is not correct"
            
            upsampled_pc, up_batch = self.upsampler(x=tm_out, pos = pred_pc_seq[-1][last_frame_idx,:],batch=encoder_batch, return_batch=True)
            pc_reconstructed = self.reconstructor(upsampled_pc)
            pc_f1 = self.refiner(x = upsampled_pc, pos = pc_reconstructed, batch=up_batch)
            assert pc_f1.shape == torch.Size([int(num_points * batch_size), 3]), "pc_f1 shape is not correct"

            
            pred_pc_seq.append(pc_f1)
            

        return pred_pc_seq



if __name__ == "__main__":
    class KittiLoader(DataLoader):
        def __init__(self, dataset, batch_size):
            super(KittiLoader, self).__init__(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        def collate_fn(self, batch):
            a_list, b_list = zip(*batch)
            a_batch = torch.cat(a_list, dim=1)
            b_batch = torch.cat(b_list, dim=1)
            return a_batch, b_batch

    encoder_args = {'channels': 32, 'k': 30, 'dilations': (1,2,3), 'n_idgcn_blocks':2, 'n_dgcn_blocks':2, 'radio': 0.25}
    encoder_kargs = { 'use_radius_graph': False, 'use_bottleneck': True, 'use_pooling': True, 'use_residual': True, 'conv': 'edge', 'pool_type': 'mean',
    'dynamic': False, 'hierarchical': True,}
    dim_args = {'global_fea_dim': 1024 }
    att_args = {'num_heads': 4, 'num_neighs': 20}
    upsampler_args = { "upsampler": "nodeshuffle", "in_channels": 64, "out_channels": 32, "k": 30, "r": 4}
    refiner_kargs = { "in_channels": 32, "out_channels": 3, "k": 30, "dilations": (1,2), "add_points": True }
    train_config = {'lr': 1e-3, 'batch_size':4, 'num_pred': 5, 'beta': (0.9, 0.999)}

    data_config = {'root': "../kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00', '01','02'], 'val_seqs': ['03']}

    pc_fc_model = PC_forecasting_model_0_0(encoder_args, encoder_kargs, dim_args, att_args, upsampler_args, refiner_kargs)

    pc_seq_dataset_tr = KittiDataset(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['val_seqs'])
    print(len(pc_seq_dataset_tr))
    k_loader_tr = KittiLoader(pc_seq_dataset_tr, batch_size=4)

    epoch = 0
    for epoch in range(5):
        print("epoch: ", epoch)
        for i, x in enumerate(k_loader_tr):
            print(i)
            input_seq, gt_seq = x
            print(input_seq.shape)
            print(gt_seq.shape)
            res_list = pc_fc_model(input_seq, 4, 5, 4096, "cpu")
            for res in res_list:
                print(res.shape)
        epoch += 1
    







