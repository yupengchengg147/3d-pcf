import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.nn import knn_graph, knn, knn_interpolate
from torch_geometric.nn import MLP


class SAModule(torch.nn.Module):
    def __init__(self, ratio, nn, r, k, knn=True):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.k = k
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.knn = knn

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)

        if self.knn:
            row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        else:
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.k)
            
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class PC_Encoder(nn.Module):
    def __init__(self, indim=3, knn=True) -> None:
        super(PC_Encoder, self).__init__()
        self.sa1 = SAModule(0.25, MLP([3+indim, 32, 32, 64]), r=0.2*8.5, k=32, knn=knn)
        self.sa2 = SAModule(0.5, MLP([64+3, 96, 96, 128]), r=0.4*8.5, k=32, knn=knn)
        self.sa3 = SAModule(0.5, MLP([128+3, 128, 128, 256]), r=0.8*8.5, k=32, knn=knn)

    def forward(self, fea, xyz, batch):
        fea1, xyz1, batch1 = self.sa1(fea, xyz, batch)
        fea2, xyz2, batch2 = self.sa2(fea1, xyz1, batch1)
        fea3, xyz3, batch3 = self.sa3(fea2, xyz2, batch2)
        return fea3, xyz3, batch3


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_prev, pos_prev, batch_prev):
        x = knn_interpolate(x, pos, pos_prev, batch, batch_prev, k=self.k)
        if x_prev is not None:
            x = torch.cat([x, x_prev], dim=1)
        x = self.nn(x)
        return x, pos_prev, batch_prev

class Local_Point_Trans(nn.Module):
    def __init__(self, k, ch_in, ch_out) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.k = int(k)
        self.lin_p = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, ch_out),
        )

        self.lin_q = nn.Linear(ch_in, ch_out)
        self.lin_k = nn.Linear(ch_in, ch_out)
        self.lin_v = nn.Linear(ch_in, ch_out)
        self.lin_w = nn.Sequential(
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(),
            nn.Linear(ch_out, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU())
        
        self.att = nn.Softmax(dim=1)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fea_i, fea_last, xyz_i, xyz_last, batch, t_i, t_last=1.0):

        _, cin = fea_i.shape
        assert cin == self.ch_in, "input channel not match"
        index = knn(x=xyz_i,y=xyz_last, k=self.k, batch_x=batch, batch_y=batch)

        idx_last, idx_i = index[0], index[1]
        fea_last = fea_last[idx_last].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]
        fea_i = fea_i[idx_i].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]


        xyzt_last = torch.cat((xyz_last[idx_last], t_last * torch.ones_like(xyz_last[idx_last][:,:1])), 
                               dim=-1).reshape(-1, self.k, 4).contiguous()
        xyzt_i = torch.cat((xyz_i[idx_i], t_i * torch.ones_like(xyz_i[idx_i][:,:1])), 
                           dim=-1).reshape(-1, self.k, 4).contiguous()

        pe = xyzt_i - xyzt_last
        for i, layer in enumerate(self.lin_p): pe = layer(pe.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pe)
        w = self.lin_q(fea_last) - self.lin_k(fea_i) + pe
        for i, layer in enumerate(self.lin_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.att(w)
        v = self.lin_v(fea_i) + pe

        res = torch.sum((w * v), dim=1)

        return res


class LPT_raw(nn.Module):
    """
    没有positional encoding, 只有gat
    """
    def __init__(self,k, ch_in, ch_out) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.k = int(k)

        self.lin_q = nn.Linear(ch_in, ch_out)
        self.lin_k = nn.Linear(ch_in, ch_out)
        self.lin_v = nn.Linear(ch_in, ch_out)
        self.lin_w = nn.Sequential(
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(),
            nn.Linear(ch_out, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU())
        
        self.att = nn.Softmax(dim=1)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fea_i, fea_last, xyz_i, xyz_last, batch):

        _, cin = fea_i.shape
        assert cin == self.ch_in, "input channel not match"

        index = knn(x=xyz_i,y=xyz_last, k=self.k, batch_x=batch, batch_y=batch)

        idx_last, idx_i = index[0], index[1]

        fea_last = fea_last[idx_last].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]
        fea_i = fea_i[idx_i].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]


        w = self.lin_q(fea_last) - self.lin_k(fea_i) 
        for i, layer in enumerate(self.lin_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.att(w)
        v = self.lin_v(fea_i) 

        res = torch.sum((w * v), dim=1)

        return res

class MS_LSTM_Cell(nn.Module):
    '''
    simple LSTM cell from MoNet
    Parameters:
        content_size: 64, 128, 256
        motion_size: 64, 128, 256
        hidden_size: 128, 256, 512
    Input:
        H0, C0 (B, hidden_size, N)
        fi: (B*N, content_size)
        fti: (B*N, motion_size)
    Output:
        H1: hidden state
        C1: cell state
    '''
    def __init__(self, f_con, f_t, hidden_size):
        super(MS_LSTM_Cell, self).__init__()
        # self.batch_size = batch_size
        self.f_con = f_con
        self.f_t = f_t
        self.feature_size = f_con + f_t
        self.hidden_size = hidden_size

        self.mlp_I = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_F = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_O = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")

        self.mlp_C0 = MLP([self.hidden_size+ self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_C1_1 = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, H0, C0, f01, f12, f1):

        gate_i = torch.sigmoid(self.mlp_I(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)
        gate_f = torch.sigmoid(self.mlp_F(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)
        gate_o = torch.sigmoid(self.mlp_O(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)

        C0_hat = self.mlp_C0(torch.cat((f01, C0), dim=-1)) #(B*N, hidden_size)
        C1_tilde = torch.tanh(self.mlp_C1_1(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)

        C1 = torch.mul(gate_f, C0_hat) + torch.mul(gate_i, C1_tilde) #(B*N, hidden_size)
        H1 = torch.mul(gate_o, torch.tanh(C1)) #(B*N, hidden_size)

        return H1, C1


class LPT_BD(nn.Module):
    def __init__(self, k, ch_in, ch_out) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.k = int(k)
        self.lin_p = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, ch_out),
        )

        self.lin_q = nn.Linear(ch_in, ch_out)
        self.lin_k = nn.Linear(ch_in, ch_out)
        self.lin_v = nn.Linear(ch_in, ch_out)
        self.lin_w = nn.Sequential(
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(),
            nn.Linear(ch_out, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU())
        
        self.att = nn.Softmax(dim=1)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fea_prev, fea_cur, fea_next, xyz_prev, xyz_cur, xyz_next, batch, t_prev, t_cur, t_next):
        f01 = self.gat_btw_fr(fea_prev, fea_cur, xyz_prev, xyz_cur, batch, t_prev, t_cur)
        f10 = self.gat_btw_fr(fea_next, fea_cur, xyz_next, xyz_cur, batch, t_next, t_cur)
        # ft_cur = torch.cat((f01, f10), dim=-1)
        return f01, f10


    def gat_btw_fr(self, fea_kv, fea_q, xyz_kv, xyz_q, batch, t_kv, t_q):

        _, cin = fea_kv.shape
        assert cin == self.ch_in, "input channel not match"
        index = knn(x=xyz_kv,y=xyz_q, k=self.k, batch_x=batch, batch_y=batch)

        idx_last, idx_i = index[0], index[1]
        fea_q = fea_q[idx_last].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]
        fea_kv = fea_kv[idx_i].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]


        xyzt_last = torch.cat((xyz_q[idx_last], t_q * torch.ones_like(xyz_q[idx_last][:,:1])), 
                                dim=-1).reshape(-1, self.k, 4).contiguous()
        xyzt_i = torch.cat((xyz_kv[idx_i], t_kv * torch.ones_like(xyz_kv[idx_i][:,:1])), 
                            dim=-1).reshape(-1, self.k, 4).contiguous()

        pe = xyzt_i - xyzt_last
        for i, layer in enumerate(self.lin_p): pe = layer(pe.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pe)
        w = self.lin_q(fea_q) - self.lin_k(fea_kv) + pe
        for i, layer in enumerate(self.lin_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.att(w)
        v = self.lin_v(fea_kv) + pe

        res = torch.sum((w * v), dim=1)

        return res
    

class LPT_BD_not(nn.Module):
    def __init__(self, k, ch_in, ch_out) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.k = int(k)
        self.lin_p = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, ch_out),
        )

        self.lin_q = nn.Linear(ch_in, ch_out)
        self.lin_k = nn.Linear(ch_in, ch_out)
        self.lin_v = nn.Linear(ch_in, ch_out)
        self.lin_w = nn.Sequential(
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(),
            nn.Linear(ch_out, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU())
        
        self.att = nn.Softmax(dim=1)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fea_prev, fea_cur, fea_next, xyz_prev, xyz_cur, xyz_next, batch):
        f01 = self.gat_btw_fr(fea_prev, fea_cur, xyz_prev, xyz_cur, batch)
        f10 = self.gat_btw_fr(fea_next, fea_cur, xyz_next, xyz_cur, batch)
        # ft_cur = torch.cat((f01, f10), dim=-1)
        return f01, f10


    def gat_btw_fr(self, fea_kv, fea_q, xyz_kv, xyz_q, batch):

        _, cin = fea_kv.shape
        assert cin == self.ch_in, "input channel not match"
        index = knn(x=xyz_kv,y=xyz_q, k=self.k, batch_x=batch, batch_y=batch)

        idx_last, idx_i = index[0], index[1]
        fea_q = fea_q[idx_last].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]
        fea_kv = fea_kv[idx_i].reshape(-1, self.k, cin).contiguous()  # [N, k, 256]


        xyzt_last =(xyz_q[idx_last]).reshape(-1, self.k, 3).contiguous()
        xyzt_i = (xyz_kv[idx_i]).reshape(-1, self.k, 3).contiguous()

        pe = xyzt_i - xyzt_last
        for i, layer in enumerate(self.lin_p): pe = layer(pe.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pe)
        w = self.lin_q(fea_q) - self.lin_k(fea_kv) + pe
        for i, layer in enumerate(self.lin_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.att(w)
        v = self.lin_v(fea_kv) + pe

        res = torch.sum((w * v), dim=1)

        return res

class MS_LSTM_Cell_NOC(nn.Module):
    '''
    without content feature
  
    '''
    def __init__(self, f_con, f_t, hidden_size):
        super(MS_LSTM_Cell_NOC, self).__init__()
        # self.batch_size = batch_size
        self.f_con = 0
        self.f_t = f_t
        self.feature_size = f_con + f_t
        self.hidden_size = hidden_size

        self.mlp_I = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_F = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_O = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")

        self.mlp_C0 = MLP([self.hidden_size+ self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_C1_1 = MLP([self.hidden_size+ self.f_con + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, H0, C0, f01, f12):

        gate_i = torch.sigmoid(self.mlp_I(torch.cat((H0, f01, f12), dim=-1))) #(B*N, hidden_size)
        gate_f = torch.sigmoid(self.mlp_F(torch.cat((H0, f01, f12), dim=-1))) #(B*N, hidden_size)
        gate_o = torch.sigmoid(self.mlp_O(torch.cat((H0, f01, f12), dim=-1))) #(B*N, hidden_size)

        C0_hat = self.mlp_C0(torch.cat((f01, C0), dim=-1)) #(B*N, hidden_size)
        C1_tilde = torch.tanh(self.mlp_C1_1(torch.cat((H0, f01, f12), dim=-1))) #(B*N, hidden_size)

        C1 = torch.mul(gate_f, C0_hat) + torch.mul(gate_i, C1_tilde) #(B*N, hidden_size)
        H1 = torch.mul(gate_o, torch.tanh(C1)) #(B*N, hidden_size)

        return H1, C1
    

class TD_Encoder(nn.Module):
    def __init__(self, dim_out) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.mlp = MLP([1, self.dim_out, self.dim_out], bias=True)
    def forward(self, td):
        return self.mlp(td)

class MS_LSTM_Cell_WITHT(nn.Module):

    def __init__(self, f_con, f_t, hidden_size, dim_t):
        super(MS_LSTM_Cell_WITHT, self).__init__()
        # self.batch_size = batch_size
        self.f_con = f_con
        self.f_t = f_t
        self.feature_size = f_con + f_t
        self.hidden_size = hidden_size

        self.te = TD_Encoder(dim_t)

        self.mlp_I = MLP([self.hidden_size+ self.f_con + dim_t + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_F = MLP([self.hidden_size+ self.f_con + dim_t + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_O = MLP([self.hidden_size+ self.f_con + dim_t + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")

        self.mlp_C0 = MLP([self.hidden_size+ self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")
        self.mlp_C1_1 = MLP([self.hidden_size+ self.f_con + dim_t + 2* self.f_t, self.hidden_size, self.hidden_size], bias=True, norm="batch_norm")

        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, H0, C0, f01, f12, f1, pos):

        td = torch.ones_like(f1[:,-1], dtype=torch.float32).unsqueeze(-1).to(f1.device) * float(pos)
        tde = self.te(td)
        f1 = torch.cat((f1, tde), dim=-1)

        gate_i = torch.sigmoid(self.mlp_I(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)
        gate_f = torch.sigmoid(self.mlp_F(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)
        gate_o = torch.sigmoid(self.mlp_O(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)

        C0_hat = self.mlp_C0(torch.cat((f01, C0), dim=-1)) #(B*N, hidden_size)
        C1_tilde = torch.tanh(self.mlp_C1_1(torch.cat((H0, f01, f12, f1), dim=-1))) #(B*N, hidden_size)

        C1 = torch.mul(gate_f, C0_hat) + torch.mul(gate_i, C1_tilde) #(B*N, hidden_size)
        H1 = torch.mul(gate_o, torch.tanh(C1)) #(B*N, hidden_size)

        return H1, C1
    



class Global_Encoder(nn.Module):

    def __init__(self, num_neighbors, mlp, witht=True) -> None:
        super().__init__()
        self.mlp = mlp
        self.num_neighbors = num_neighbors

        self.witht = witht
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, num_samples, f1_list, xyz1_list, dbatch):
        fea_source, fea_tar = self.wighted_sample(num_samples, f1_list, xyz1_list, dbatch)
        tmp = torch.cat([fea_tar, fea_tar-fea_source], dim=2)
        out = torch.max(self.mlp(tmp), dim=-2)[0]
        return torch.tanh(out)

    def wighted_sample(self, num_samples, f1_list, xyz1_list, dbatch):
        #input: xyz1_list, f1_list
        l = len(f1_list)
        n, h = f1_list[-1].shape

        fea_set = torch.zeros(l-1, n, self.num_neighbors, h).to(f1_list[-1].device)
        delta_r_set = torch.zeros(l-1, n, self.num_neighbors).to(f1_list[-1].device)
        
        anchor = xyz1_list[-1]
        for i in range(0, l-1):
            idx_anchor, idx_i= knn(x=xyz1_list[i],y=anchor, k=self.num_neighbors, batch_x=dbatch, batch_y=dbatch)
            # 每个frame先找 num_neighbors 个点

            delta_xyz = (anchor[idx_anchor,:] - xyz1_list[i][idx_i,:]).reshape(-1, self.num_neighbors, 3).contiguous()
            delta_r = torch.sqrt(torch.sum(delta_xyz**2, dim=2))
            delta_r_set[i,:,:] = delta_r
            #可能会用index_operation的问题，待定

            fea_set[i,:,:] = f1_list[i][idx_i,:].reshape(-1, self.num_neighbors, h).contiguous()

        fea_set = fea_set.view(1, n, -1, h).contiguous().squeeze(0) #[B*N, num_neighbors, h]
        delta_r_set = delta_r_set.view(n, -1).contiguous() 

        #根据delta_r采样， P 正比于 exp(-belta * delta_r)
        alpha, belta = 0.5 ,0.5
        # belta越小，分布越平滑, 
        # belta太大的话，跟第一步直接取num_samples个点的效果差不多

        pr_r = torch.softmax(delta_r_set* (-1.*belta), dim=1) # [B*N, num_neighbors * (T-1)]

        if self.witht:
            delta_t = torch.repeat_interleave(torch.arange(l-1, 0, -1), self.num_neighbors).to(f1_list[-1].device)
            delta_t_set = delta_t.unsqueeze(0).repeat(n, 1).contiguous()
            pr_t = torch.softmax(delta_t_set* (-1.*0.5), dim=1) # [B*N, num_neighbors * (T-1)]
            p = 0.5 * pr_r + 0.5 * pr_t
        else:
            p = pr_r
        
        idx = torch.multinomial(p, num_samples, replacement=False) # [B*N, num_samples * (T-1)]

        fea_source = fea_set[torch.arange(n).unsqueeze(1),idx,:] # [B*N, num_samples * (T-1), 64]
        _, sum_n, _ = fea_source.shape
        fea_tar = f1_list[-1].unsqueeze(1).repeat(1, sum_n, 1) # [B*N, num_samples * (T-1), 64]
    
        return fea_source, fea_tar



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
    



