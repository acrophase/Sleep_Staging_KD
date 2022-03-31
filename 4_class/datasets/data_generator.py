import random
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.utils import class_weight

class DataGenerator(pl.LightningDataModule):
    def __init__(self,
        batch_size=256,
        aasm_data_dir = None,
        rk_data_dir= None,
        n_workers = 4,
        *args,
        **kwargs,):
        super().__init__()
        self.batch_size = batch.size
        self.aasm_data_dir = aasm_data_dir
        self.rk_data_dir = rk_data_dir
        self.n_workers= n_workers
        self.on_epoch_end()
        
    def on_epoch_end(self):
        # updates index after each epoch
        pass
    
    def __getitem__(self, index):
        # generate 1 batch of data
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # X, y = self.__data_generation(<iterator of data)
        # return X, y
        pass
    
    def __len__(self):
        # defines no. of batches
        return self.n // self.batch_size

    def __datageneration(self):
        ## orivate method to generate data
        pass
