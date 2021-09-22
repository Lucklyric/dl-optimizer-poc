import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from mrc_insar_common.util.pt.init import weight_init

logger = logging.getLogger(__name__)


class POCFc(pl.LightningModule):
    def __init__(self,
                 lr=0.0003,
                 num_layers=7,
                 num_features=64,
                 *args,
                 **kwargs):
        super().__init__()
        self.lr = lr
        layers = []
        layers.append(
            nn.Linear(2,num_features))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(
                nn.Linear(num_features, num_features))
            layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(
            nn.Linear(num_features, 2))
        self.net = nn.Sequential(*layers)

        self.net.apply(weight_init)

    def forward(self, x):
        x = self.net(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        observered_in = batch['input']
        label_out = batch['output']


        # run forward
        out = self.forward(observered_in)

        x1_loss = torch.mean((out[:, 0] - label_out[:, 0])**2)
        x2_loss = torch.mean((out[:, 1] - label_out[:, 1])**2)
        loss = x1_loss + x2_loss

        self.log('my_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.log('x1_loss',
                 x1_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.log('x2_loss',
                 x2_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, x, batch_idx):
        observered_in = x['input']
        label_out = x['output']
        with torch.no_grad():
            out = self.forward(observered_in)

            x1_loss = torch.mean((out[:, 0] - label_out[:, 0])**2)
            x2_loss = torch.mean((out[:, 1] - label_out[:, 1])**2)

            metric_dict = {'val_x1_error': x1_loss, 'val_x2_error': x2_loss}

            logger.info(metric_dict)

            self.log_dict(metric_dict)

    def test_step(self, x, batch_idx):
        observered_in = x['input']
        label_out = x['output']
        with torch.no_grad():
            out = self.forward(observered_in)

            x1_loss = torch.mean((out[:, 0] - label_out[:, 0])**2)
            x2_loss = torch.mean((out[:, 1] - label_out[:, 1])**2)

            metric_dict = {'val_x1_error': x1_loss, 'val_x2_error': x2_loss}

            logger.info(metric_dict)

            self.log_dict(metric_dict)
        return (x1_loss + x2_loss).detach().cpu().numpy()

    def test_epoch_end(self, outputs):
        # for test_epoch_end showcase, we record batch mse and batch std
        all_res = np.asarray(outputs)
        self.log_dict({'b_mse': np.mean(all_res), 'b_std': np.std(all_res)})
