# 3行都有，f1那行加time difference, 使用cat，把time difference 投影为16dim，cat到feature上

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mo_lstm_content_t import PC_MO_LSTM_TDE
from dataset.kitti_dataset_v2 import KittiDataset_2

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from torch.optim.lr_scheduler import StepLR

import random
from monet_loss import cd_list

def set_seed(seed):
    '''
    Set random seed for torch, numpy and python
    '''
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.benchmark=False 
    torch.backends.cudnn.deterministic=True


class MO_LSTM_TDE(pl.LightningModule):
    def __init__(self, train_config):
        super(MO_LSTM_TDE, self).__init__()
        self.model = PC_MO_LSTM_TDE()
        self.loss_fn = cd_list
        self.lr = train_config['lr']
        self.beta = train_config['beta']
        self.batch_size = int(train_config['batch_size'])
        self.num_pred = int(train_config['num_pred'])
    
    def forward(self, input_pc_seq):
        return self.model(input_pc_seq, self.num_pred, self.device)
    
    def training_step(self, batch, batch_idx):
        input_pc_seq, gt_pc_seq = batch 
        detail_list = self.forward(input_pc_seq) 
        loss = self.loss_fn(detail_list, gt_pc_seq, self.batch_size)
        
        print("tain_loss: ", loss)
        self.log('train_loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_pc_seq, gt_pc_seq = batch 
        detail_list = self.forward(input_pc_seq) 

        loss = self.loss_fn(detail_list, gt_pc_seq, self.batch_size)

        self.log('val_loss', loss, sync_dist=True)
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
    
    train_config = {'lr': 1e-3, 'batch_size':6, 'num_pred': 5, 'beta': (0.9, 0.999), 'num_points': 16384}
    data_config = {'root': "/usr/stud/ype/storage/user/kittidata/dataset/sequences", 'npoints': 16384, 'input_num': 5, 'pred_num': 5, 'tr_seqs': ['00', '01','02','03','04','05'],'val_seqs': ['08','09','10']}
    
    train_dataset = KittiDataset_2(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['tr_seqs'])
    val_dataset = KittiDataset_2(root=data_config['root'], npoints=data_config['npoints'], input_num=data_config['input_num'], pred_num=data_config['pred_num'], seqs=data_config['val_seqs'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    pc_fc_model = MO_LSTM_TDE(train_config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/mo_lstm_tde_0019/',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
        every_n_epochs=1,
    )

    seq_logger = TensorBoardLogger("logs", name="molstm_tde_0019")

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, num_nodes=1,
        callbacks=[checkpoint_callback],
        max_epochs=5000,
        min_epochs=200,
        logger=seq_logger,
        log_every_n_steps=1
    )


    trainer.fit(pc_fc_model, train_dataloader, val_dataloader)

if __name__ == "__main__":

    set_seed(147)
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    train_model()

