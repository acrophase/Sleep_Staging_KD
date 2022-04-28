import random
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
import os
import glob
import random
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
import os

class MASSDataset(Dataset):
    def __init__(self, eeg_path_lists, ecg_path_lists, slp_stg_path_lists):
        self.eeg_path_lists = eeg_path_lists
        self.ecg_path_lists = ecg_path_lists
        self.slp_stg_path_lists = slp_stg_path_lists # self.path_lists is list of file paths (i.e., aasm path and rk path)
        self.len_list = [0]
        # self.iter = 0
        len_sum = 0
        self.eeg_file_path_list = []
        self.ecg_file_path_list = []
        self.slp_stg_file_path_list = []
        self.eeg_data = self.ecg_data = self.slp_stg_data = torch.Tensor([])
        self.file_path_iter = 0
        for i in range(len(self.eeg_path_lists)):
            eeg_file_path = sorted(os.listdir(self.eeg_path_lists[i]))
            ecg_file_path = sorted(os.listdir(self.ecg_path_lists[i]))
            slp_stg_file_path = sorted(os.listdir(self.slp_stg_path_lists[i]))
            len_sum+=len(eeg_file_path)
            self.len_list.append(len_sum)
            self.eeg_file_path_list.append(eeg_file_path)
            self.ecg_file_path_list.append(ecg_file_path)
            self.slp_stg_file_path_list.append(slp_stg_file_path)
        # self.len_list is list of accumulated lengths of files in each file path, with first value as 0
        # self.file_path_list is list of list of files under each file path in self.path_lists
    
    def __len__(self):
        return self.len_list[-1]
    # returns total length of files = last value in list of accumulated length of files

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if  self.file_path_iter < len(self.len_list)-2:   
            if idx == self.len_list[self.file_path_iter+1]:
                self.file_path_iter+=1

        idx_1 = idx - self.len_list[self.file_path_iter]
                   
        # import pdb; pdb.set_trace()
        eeg_path = os.path.join(self.eeg_path_lists[self.file_path_iter], self.eeg_file_path_list[self.file_path_iter][idx_1])
        ecg_path = os.path.join(self.ecg_path_lists[self.file_path_iter], self.ecg_file_path_list[self.file_path_iter][idx_1])
        slp_stg_path = os.path.join(self.slp_stg_path_lists[self.file_path_iter], self.slp_stg_file_path_list[self.file_path_iter][idx_1])
        
        tens_eeg = torch.from_numpy(np.genfromtxt(eeg_path, delimiter=","))
        tens_ecg = torch.from_numpy(np.genfromtxt(ecg_path, delimiter=","))
        tens_slp_stg = torch.from_numpy(np.genfromtxt(slp_stg_path, delimiter=","))
        eeg_file_data = torch.as_tensor(tens_eeg)
        ecg_file_data = torch.as_tensor(tens_ecg)
        slp_stg_file_data = torch.as_tensor(tens_slp_stg)
        
        return eeg_file_data, ecg_file_data, slp_stg_file_data

        # self.eeg_data = torch.cat([self.eeg_data,eeg_file_data], dim=0)
    
        
        # if 'ECG' in self.path_lists[self.file_path_iter]:
        #     ecg_path = os.path.join(self.path_lists[self.file_path_iter], self.file_path_list[self.file_path_iter][idx])
        #     tens = torch.from_numpy(np.genfromtxt(ecg_path, delimiter=","))
        #     ecg_file_data = torch.as_tensor(tens)
        #     self.ecg_data = torch.cat([self.ecg_data,ecg_file_data], dim=0)
        # if 'Sleep_stages' in self.path_lists[self.file_path_iter]:
        #     slp_stg_path = os.path.join(self.path_lists[self.file_path_iter], self.file_path_list[self.file_path_iter][idx])
        #     tens = torch.from_numpy(np.genfromtxt(slp_stg_path, delimiter=","))
        #     slp_stg_file_data = torch.as_tensor(tens)
        #     self.slp_stg_data = torch.cat([self.slp_stg_data,slp_stg_file_data], dim=0)
        # print(self.file_path_list[self.file_path_iter][idx])
        # # import pdb; pdb.set_trace()
        # if len(self.eeg_data) == 256*(len(self.file_path_list[0])+ len(self.file_path_list[1])) \
        # and len(self.ecg_data) == 256*(len(self.file_path_list[2])+ len(self.file_path_list[3])) \
        # and len(self.slp_stg_data) == 256*(len(self.file_path_list[4])+ len(self.file_path_list[5])):
        #     import pdb; pdb.set_trace()
        #     return_data = torch.column_stack([self.eeg_data, self.ecg_data, self.slp_stg_data])
        #     return_data = torch.Tensor(return_data)
        #     return return_data


class MASSBatchSmallDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        self.n_workers = 32
        iter_aasm_eeg_train = iter_rk_eeg_train = iter_aasm_ecg_train = iter_rk_ecg_train = iter_aasm_slp_stg_train = iter_rk_slp_stg_train = 0
        iter_aasm_eeg_val = iter_rk_eeg_val = iter_aasm_ecg_val = iter_rk_ecg_val = iter_aasm_slp_stg_val = iter_rk_slp_stg_val = 0
        iter_aasm_eeg_test = iter_rk_eeg_test = iter_aasm_ecg_test = iter_rk_ecg_test = iter_aasm_slp_stg_test = iter_rk_slp_stg_test = 0
        self.eeg_train = self.eeg_val = self.eeg_test = torch.Tensor([])
        self.ecg_train = self.ecg_val = self.ecg_test = torch.Tensor([])
        self.slp_stg_train = self.slp_stg_val = self.slp_stg_test = torch.Tensor([])
        # import pdb; pdb.set_trace()
        self.train_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/train/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/train/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/train/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/train/'])
        self.val_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/val/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/val/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/val/'])
        self.test_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/test/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/test/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/test/'])
        
        # for idx, tens in enumerate(self.train_dataset):
        #     pass# import pdb; pdb.set_trace()
        # # self.val_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/val/'])
        # self.test_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/test/'])
        # In AASM, there are train, val, and test files each for EEG, ECG and Sleep Stages. Similarly for RK
        # Train EEG-ECG-Sleep stages for AASM and RK combined and given as input to train dataloader. Similarly for val and test.
        # self.eeg_train_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/train/'])
        # self.ecg_train_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/train/'])
        # self.slp_stg_train_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/train/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/train/'])
        
        # self.eeg_val_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/val/'])
        # self.ecg_val_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/val/'])
        # self.slp_stg_val_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/val/'])

        # self.eeg_test_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/test/'])
        # self.ecg_test_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/test/'])
        # self.slp_stg_test_dataset = MASSDataset(['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/test/'])

    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,num_workers=self.n_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,num_workers=self.n_workers, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,num_workers=self.n_workers, pin_memory=True)


# mass_dataset = MASSDataset('/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Biosig_files')
# train_loader = DataLoader(mass_dataset)

# for idx, tensor in enumerate(train_loader):
#     eeg_sample = tensor
#     print("mean is", eeg_sample.mean())
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        from argparse import ArgumentParser

        # DATALOADER specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # dataset_group = parser.add_argument_group("dataset")
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=256, type=int)
        dataloader_group.add_argument("--n_workers", default=32, type=int)

        return parser
if __name__ == "__main__":
    seed_everything(0, workers=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    mdm = MASSBatchSmallDataModule()

