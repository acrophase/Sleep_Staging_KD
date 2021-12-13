import random
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.utils import class_weight


class MassDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size=256,
        aasm_data_dir = None,
        rk_data_dir= None,
        n_workers=32,
        *args,
        **kwargs,
    ):
        
        super().__init__()
        self.batch_size = batch_size
        self.aasm_data_dir = aasm_data_dir
        self.rk_data_dir = rk_data_dir
        self.n_workers= n_workers
        
    def setup(self, stage = None):
        if stage == "fit":
            aasm_filepath = self.aasm_data_dir
            rk_filepath = self.rk_data_dir

            # import pdb;pdb.set_trace()
            self.aasm_train_data = torch.load(aasm_filepath + 'eeg_ecg_1ch_train.pt')
            self.aasm_train_slp_stg = torch.load(aasm_filepath + 'slp_stg_train_lbl.pt')
            self.rk_train_data = torch.load(rk_filepath + 'eeg_ecg_1ch_train.pt')
            self.rk_train_slp_stg = torch.load(rk_filepath + 'slp_stg_train_lbl.pt')
            
            self.eeg_train_data = torch.cat([self.aasm_train_data[0],self.rk_train_data[0]],dim=0)
            self.ecg_train_data = torch.cat([self.aasm_train_data[1],self.rk_train_data[1]],dim=0)
            self.train_slp_stg =  torch.cat([self.aasm_train_slp_stg,self.rk_train_slp_stg],dim=0)
            # Calculating Class_weights to use for weighted cross entropy loss
            y=self.train_slp_stg.cpu()
            self.train_class_weights = class_weight.compute_class_weight('balanced',np.arange(4),y.numpy().ravel())
            self.train_class_weights = torch.tensor(self.train_class_weights,dtype=torch.float).cuda()
            
            self.training_data = TensorDataset(
                self.eeg_train_data,
                self.ecg_train_data,
                self.train_slp_stg
                )

            self.aasm_val_data = torch.load(aasm_filepath + 'eeg_ecg_1ch_eval.pt')
            self.aasm_val_slp_stg = torch.load(aasm_filepath + 'slp_stg_eval_lbl.pt')
            self.rk_val_data = torch.load(rk_filepath + 'eeg_ecg_1ch_eval.pt')
            self.rk_val_slp_stg = torch.load(rk_filepath + 'slp_stg_eval_lbl.pt')

            self.eeg_val_data = torch.cat([self.aasm_val_data[0],self.rk_val_data[0]],dim=0)
            self.ecg_val_data = torch.cat([self.aasm_val_data[1],self.rk_val_data[1]],dim=0)
            self.val_slp_stg =  torch.cat([self.aasm_val_slp_stg,self.rk_val_slp_stg],dim=0)

            self.validation_data = TensorDataset(
                self.eeg_val_data,
                self.ecg_val_data,
                self.val_slp_stg
                )

        elif stage == "test":

            aasm_filepath = self.aasm_data_dir
            rk_filepath = self.rk_data_dir

            self.aasm_test_data = torch.load(aasm_filepath + 'eeg_ecg_1ch_test.pt')
            self.aasm_test_slp_stg = torch.load(aasm_filepath + 'slp_stg_test_lbl.pt')
            self.rk_test_data = torch.load(rk_filepath + 'eeg_ecg_1ch_test.pt')
            self.rk_test_slp_stg = torch.load(rk_filepath + 'slp_stg_test_lbl.pt')

            self.eeg_test_data = torch.cat([self.aasm_test_data[0],self.rk_test_data[0]],dim=0)
            self.ecg_test_data = torch.cat([self.aasm_test_data[1],self.rk_test_data[1]],dim=0)
            self.test_slp_stg =  torch.cat([self.aasm_test_slp_stg,self.rk_test_slp_stg],dim=0)
            
            self.testing_data = TensorDataset(
                self.eeg_test_data,
                self.ecg_test_data,
                self.test_slp_stg
                )
    def train_dataloader(self):
        """Return training dataloader."""
        # self.setup("fit")
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        # self.setup("fit")
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        # self.setup("test")
        return DataLoader(
            self.testing_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.n_workers
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        from argparse import ArgumentParser

        # DATASET specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("--aasm_data_dir", default="/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch_subjectwisesplit/4class/ALL_DATA/AASM/", help= 'Enter path to data PT files')
        dataset_group.add_argument("--rk_data_dir", default="/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch_subjectwisesplit/4class/ALL_DATA/R_K/30s/", help= 'Enter path to data PT files')

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=256, type=int)
        dataloader_group.add_argument("--n_workers", default=32, type=int)

        return parser

if __name__ == "__main__":

    seed_everything(42, workers=True)

    np.random.seed(42)
    random.seed(42)

    dm_params = dict(
        batch_size=256,
        # aasm_data_dir='/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/POCT/4_class/AASM/',
        # rk_data_dir='/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/POCT/4_class/R_K_30s/',
        n_workers=32,
    )

    dm = MassDataModule(**dm_params)
    dm.setup("fit")
    dm.setup("test")
