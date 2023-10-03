import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn_graph, knn
from torch_geometric.nn import global_max_pool, global_mean_pool
from utils.losses import chamfer_dist, chamfer_dist_repulsion, density_chamfer_dist
from models.model_knn_not import PC_forecasting_model_knn_not


from dataset.kitti_dataset_v2 import KittiDataset_2

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from torch.optim.lr_scheduler import StepLR



def chamfer_dist_btw_seq(predicted_seq, gt_seq, batch_size):

    seq_loss = 0
    m_cd = 0

    for i, pc in enumerate(predicted_seq):
        pc = pc.reshape(batch_size, -1, 3).contiguous()
        gt = gt_seq[i].permute(0,2,1).contiguous()

        # print("pc.shape: ", pc.shape)
        # print("gt.shape: ", gt.shape)

        # gt: (B*N, 3) -- (B, N, 3)
        # pc: (B*N, 3) -- (B, N, 3)
        loss, _, cd_t = density_chamfer_dist(pc,gt)

        m_cd += cd_t.mean()
        seq_loss += loss.mean()
    
    seq_loss = seq_loss / len(predicted_seq)
    m_cd = m_cd / len(predicted_seq)

    return seq_loss, m_cd


class PointCloudForecastingModel_knn_not(pl.LightningModule):
    def __init__(self, encoder_args, att_args, upsampler_args, refiner_kargs, train_config):
        super(PointCloudForecastingModel_knn_not, self).__init__()
        self.model = PC_forecasting_model_knn_not(encoder_args, att_args, upsampler_args, refiner_kargs, train_config['num_points'])

        self.loss_fn = chamfer_dist_btw_seq
        self.lr = train_config['lr']
        self.beta = train_config['beta']
        self.batch_size = int(train_config['batch_size'])
        self.num_pred = int(train_config['num_pred'])
        self.num_points = int(train_config['num_points'])
    
    def forward(self, input_pc_seq):
        return self.model(input_pc_seq, self.num_pred, self.device)
    
    def training_step(self, batch, batch_idx):
        input_pc_seq,_, gt_pc_seq = batch # list of [(B, 3, N),]
        coarse_list, detail_list = self.forward(input_pc_seq) # a list of (B*N, 3)

        _, loss_c = self.loss_fn(coarse_list, gt_pc_seq, self.batch_size)
        _ , loss_d = self.loss_fn(detail_list, gt_pc_seq, self.batch_size)
        
        train_step = self.global_step
        if train_step < 1500:
            alpha = 0.1
        elif train_step < 3000:
            alpha = 0.5
        else:
            alpha = 1.0

        # loss = loss_c + alpha * loss_d
        loss = loss_d

        print("tain_loss_c: ", loss_c)
        print("tain_loss_d: ", loss_d)
        print("tain_loss: ", loss)
        self.log('train_loss', loss_d)

        return loss

    def validation_step(self, batch, batch_idx):
        input_pc_seq, _, gt_pc_seq = batch 
        coarse_list, detail_list = self.forward(input_pc_seq) 

        _ , loss = self.loss_fn(detail_list, gt_pc_seq, self.batch_size)

        self.log('val_loss', loss)
        print("val_cd: ",loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.beta)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=800, gamma=0.5),
            'interval': 'epoch',  # 或 'step'，取决于你希望在每个 epoch 或每个训练步骤之后更新学习率
        }
        return [optimizer], [scheduler]
        # return optimizer
    

def train_model():
    encoder_args = {'df':3}
    att_args = {'num_heads': 4, 'num_neighs': 20}
    upsampler_args = {
        "upsampler": "nodeshuffle",
        "in_channels": 256,
        "out_channels": 64,
        "k": 40,
        "r": 16, 
        "conv":"gatv2",
        "heads":4
    }

    refiner_kargs = {
        "in_channels": 64,
        "out_channels": 3,
        "k": 30,
        "dilations": (1,2),
        "add_points": True
    }

    train_config = {'lr': 0.5*1e-4, 'batch_size':6, 'num_pred': 1, 'beta': (0.9, 0.999), 'num_points': 4096}

    data_config = {'root': "/home/stud/ding/PC_FC/PC_forecasting/kittiraw/dataset/sequences", 'npoints': 4096, 'input_num': 5, 'pred_num': 1, 'tr_seqs': ['00', '01','02','03','04','05','06','07'],'val_seqs': ['08','09','10']}
    train_dataset = KittiDataset_2(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['tr_seqs'])
    val_dataset = KittiDataset_2(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['val_seqs'])
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], drop_last=True)

    # pc_fc_model = PointCloudForecastingModel_knn_not(encoder_args, att_args, upsampler_args, refiner_kargs, train_config)
    pc_fc_model = PointCloudForecastingModel_knn_not.load_from_checkpoint(checkpoint_path='checkpoints/knn_not/best-checkpoint-epoch=42-val_loss=9.35.ckpt', 
                                                                          encoder_args=encoder_args, att_args=att_args, upsampler_args=upsampler_args, refiner_kargs=refiner_kargs, train_config=train_config)
    
    for name, param in pc_fc_model.named_parameters():
        print(name, param.shape)
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/knn_not/',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
        every_n_epochs=1,
    )



    seq_logger = TensorBoardLogger("logs", name="model_knn_not_0942")

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        callbacks=[checkpoint_callback],
        max_epochs=5000,
        min_epochs=1000,
        logger=seq_logger,
        log_every_n_steps=1
    )


    trainer.fit(pc_fc_model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    train_model()


