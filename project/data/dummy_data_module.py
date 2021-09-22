import logging

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class POCDataset(torch.utils.data.Dataset):

    def __init__(self, npz_path, data_range, *args, **kwargs):
        self.data = np.load(npz_path)['db'][data_range[0]:data_range[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        return {
            'input': entry[:2].astype(np.float32),
            'output': entry[2:].astype(np.float32)
        }


class POCDataModule(LightningDataModule):

    def __init__(self,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1,
                 train_num_workers=4,
                 val_num_workers=4,
                 test_num_workers=4,
                 data_dir='',
                 train=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

        if (train):
            self.train_dataset = POCDataset(data_dir, [0, 8000])
            self.val_dataset = POCDataset(data_dir, [8000, 9000])
            logger.info(
                f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}'
            )
        else:
            self.test_dataset = POCDataset(data_dir, [9000, 10000])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return test_loader
