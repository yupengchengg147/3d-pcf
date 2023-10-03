from typing import Any, Optional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph, knn
from torch_geometric.nn import global_max_pool, global_mean_pool
from utils.losses import chamfer_dist, chamfer_dist_repulsion, density_chamfer_dist
import time

from models.time_series_models import PC_forecasting_model_0_0
from dataset.kitti_dataset import KittiDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
# from lightning.pytorch.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch.nn.init as init



def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        init.constant_(m.bias.data, 0)




class KittiLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # Set the collate_fn to our custom collate function, overriding any user-supplied collate_fn
        kwargs['collate_fn'] = self.collate_fn
        super(KittiLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        a_list, b_list = zip(*batch)
        a_batch = torch.cat(a_list, dim=1)
        b_batch = torch.cat(b_list, dim=1)
        return a_batch, b_batch



def chamfer_dist_btw_seq(predicted_seq, gt_seq, batch_size):
    # n = int(len(predicted_seq))
    seq_loss = torch.tensor(0.0, requires_grad=True).to(predicted_seq[-1].device)
    m_cd = torch.tensor(0.0, requires_grad=False).to(predicted_seq[-1].device)
    for i, pc in enumerate(predicted_seq):
        pc = pc.reshape(batch_size, -1, 3)
        gt = gt_seq[i,:,:].reshape(batch_size, -1, 3)
        # gt: (B*N, 3) -- (B, N, 3)
        # pc: (B*N, 3) -- (B, N, 3)
        loss, _, cd_t = density_chamfer_dist(pc,gt)

        m_cd += cd_t.mean()
        seq_loss += loss.mean()
        
    return seq_loss/5, m_cd/5



class PointCloudForecastingModel(pl.LightningModule):
    def __init__(self, encoder_args, encoder_kargs, dim_args, att_args, upsampler_args, refiner_kargs, train_config):
        super(PointCloudForecastingModel, self).__init__()
        self.model = PC_forecasting_model_0_0(encoder_args, encoder_kargs, dim_args, att_args, upsampler_args, refiner_kargs)
        self.loss_fn = chamfer_dist_btw_seq
        self.lr = train_config['lr']
        self.beta = train_config['beta']
        self.batch_size = int(train_config['batch_size'])
        self.num_pred = int(train_config['num_pred'])
        self.num_points = int(train_config['num_points'])
    
    def forward(self, input_pc_seq):
        return self.model(input_pc_seq, self.batch_size, self.num_pred, self.num_points, self.device)
    
    def training_step(self, batch, batch_idx):
        input_pc_seq, gt_pc_seq = batch # (T, B*N, 3)
        pred_pc_seq = self.forward(input_pc_seq) # a list of (B*N, 3)
        loss , _ = self.loss_fn(pred_pc_seq, gt_pc_seq, self.batch_size)
        # print(loss)
        print("tain_loss: ", loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_pc_seq, gt_pc_seq = batch # (T, B*N, 3)
        pred_pc_seq = self.forward(input_pc_seq) # a list of (B*N, 3)
        val_loss, cd = self.loss_fn(pred_pc_seq, gt_pc_seq, self.batch_size)
        self.log('val_loss', val_loss)
        self.log('val_cd', cd)
        print("val_loss: ",val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.beta)
        return optimizer
    
class MyPrintingCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.train_start_t = time.time()
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("train batch time: ", time.time() - self.train_start_t)
        print("Training is ending")

    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} took {time.time() - self.epoch_start_time} seconds.")


def train_model():
    encoder_args = {'channels': 32, 'k': 30, 'dilations': (1,2,3), 'n_idgcn_blocks':2, 'n_dgcn_blocks':2, 'radio': 0.25}
    encoder_kargs = { 'use_radius_graph': False, 'use_bottleneck': True, 'use_pooling': True, 'use_residual': True, 'conv': 'edge', 'pool_type': 'mean',
    'dynamic': False, 'hierarchical': True,}
    dim_args = {'global_fea_dim': 1024 }
    att_args = {'num_heads': 4, 'num_neighs': 40}
    upsampler_args = { "upsampler": "nodeshuffle", "in_channels": 64, "out_channels": 32, "k": 30, "r": 4}
    refiner_kargs = { "in_channels": 32, "out_channels": 3, "k": 30, "dilations": (1,2), "add_points": True }
    
    train_config = {'lr': 1e-3, 'batch_size':4, 'num_pred': 5, 'beta': (0.9, 0.999), 'num_points': 4096}

    pc_fc_model = PointCloudForecastingModel(encoder_args, encoder_kargs, dim_args, att_args, upsampler_args, refiner_kargs, train_config)

    for name, param in pc_fc_model.named_parameters():
        print(name, param.shape)
    
    pc_fc_model.apply(initialize_weights)

    
    # dataset and data loader
    # data_config = {'root': "/home/wiss/ma/Pengcheng/PC_Forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00', '01','02'], 'val_seqs': ['03']}
    # data_config = {'root': "/home/wiss/ma/Pengcheng/PC_Forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['147'], 'val_seqs': ['147']}
    data_config = {'root': "/home/stud/ding/PC_FC/PC_forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00', '01','02','03','04','05','06','07'], 'val_seqs': ['08','09','10']}
    pc_seq_dataset_tr = KittiDataset(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['tr_seqs'])
    pc_seq_dataset_val = KittiDataset(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['val_seqs'])
    print("training dataset size: ", len(pc_seq_dataset_tr), "val dataset size",len(pc_seq_dataset_val))

    assert (data_config['npoints'] == train_config['num_points']),"npoints in data_config and train_config should be the same"

    k_loader_tr = KittiLoader(pc_seq_dataset_tr, batch_size=train_config['batch_size'], num_workers=32, shuffle = False, drop_last = True)
    k_loader_val = KittiLoader(pc_seq_dataset_val, batch_size=train_config['batch_size'], num_workers=32, shuffle = False, drop_last = True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        every_n_epochs=10,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
    )
    device_stats = DeviceStatsMonitor()

    seq_logger = TensorBoardLogger("logs", name="model_b6")

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        precision=32,
        callbacks=[checkpoint_callback, MyPrintingCallback()],
        max_epochs=90000,
        min_epochs=50000,
        logger=seq_logger,
        log_every_n_steps=1
    )
            # precision=16,
            # early_stop_callback

    trainer.fit(pc_fc_model, k_loader_tr, k_loader_val)

if __name__ == "__main__":
    train_model()


