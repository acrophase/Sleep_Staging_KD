import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import glob
import random
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch

class MASSDataset(Dataset):
    def __init__(self, eeg_path_lists, ecg_path_lists, slp_stg_path_lists):
        self.eeg_path_list_dir = eeg_path_lists
        self.ecg_path_list_dir = ecg_path_lists
        self.slp_stg_path_list_dir = slp_stg_path_lists # self.path_lists is list of file paths (i.e., aasm path and rk path)
        self.eeg_path_lists = []
        self.ecg_path_lists = []
        self.slp_stg_path_lists = []
        self.eeg_name = 0
        self.ecg_name = 0
        self.slp_stg_name = 0
        for j in range(len(eeg_path_lists)):
            # import pdb; pdb.set_trace()
            for k in range(len(sorted(os.listdir(self.eeg_path_list_dir[j])))):
                self.eeg_path_lists.append(sorted(os.listdir(self.eeg_path_list_dir[j]))[k])
        for j in range(len(ecg_path_lists)):
            for k in range(len(sorted(os.listdir(self.ecg_path_list_dir[j])))):
                self.ecg_path_lists.append(sorted(os.listdir(self.ecg_path_list_dir[j]))[k])
        for j in range(len(slp_stg_path_lists)):
            for k in range(len(sorted(os.listdir(self.slp_stg_path_list_dir[j])))):
                self.slp_stg_path_lists.append(sorted(os.listdir(self.slp_stg_path_list_dir[j]))[k])
        self.len_list = [0]
        # self.iter = 0
        # len_sum = 0
        # self.eeg_file_path_list = []
        # self.ecg_file_path_list = []
        # # self.slp_stg_file_path_list = []
        # self.eeg_data  = torch.Tensor([])
        # self.ecg_data
        # self.slp_stg_data = torch.Tensor([])
        # self.file_path_iter = 0
        # for i in range(len(self.eeg_path_lists)):
        # self.eeg_file_path_list = sorted(os.listdir(self.eeg_path_lists))
        # self.ecg_file_path_list = sorted(os.listdir(self.ecg_path_lists))
        # self.slp_stg_file_path_list = sorted(os.listdir(self.slp_stg_path_lists))
        # len_sum+=len(eeg_file_path) # = no. of batch files in EEG (AASM + RK) = for ECG = Sleep Stage
        # self.len_list.append(len_sum) # list of accumulated lengths of files in each file path, with first value as 0
        # self.eeg_file_path_list.append(eeg_file_path) # list of list of files under each file path
        # self.ecg_file_path_list.append(ecg_file_path)
        # self.slp_stg_file_path_list.append(slp_stg_file_path)
    
    def __len__(self):
        return len(self.eeg_path_lists)
    # returns total length of files = last value in list of accumulated length of files

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # import pdb; pdb.set_trace()
        if idx >= len(self.eeg_path_lists):
            # print("Index exceeded no. of files!!!!")
            raise ValueError("Index exceeded no. of files!!!!")
            # import pdb; pdb.set_trace()
            
        # if self.file_path_iter < len(self.len_list)-2:   
        #     if idx == self.len_list[self.file_path_iter+1]:
        #         self.file_path_iter+=1
        # import pdb; pdb.set_trace()
        # idx_1 = idx - self.len_list[self.file_path_iter]
                   
        # import pdb; pdb.set_trace()
        # try:

            # eeg_path = os.path.join(self.eeg_path_lists[self.file_path_iter], self.eeg_file_path_list[self.file_path_iter][idx_1])
            # ecg_path = os.path.join(self.ecg_path_lists[self.file_path_iter], self.ecg_file_path_list[self.file_path_iter][idx_1])
            # slp_stg_path = os.path.join(self.slp_stg_path_lists[self.file_path_iter], self.slp_stg_file_path_list[self.file_path_iter][idx_1])
            # eeg_path = os.path.join(self.eeg_path_lists, self.eeg_file_path_list[idx])
            # ecg_path = os.path.join(self.ecg_path_lists, self.ecg_file_path_list[idx])
            # slp_stg_path = os.path.join(self.slp_stg_path_lists, self.slp_stg_file_path_list[idx])
            # import pdb; pdb.set_trace()
        aasm_rk_name = 0 # 0 for AASM, 1 for RK
        if idx < len(sorted(os.listdir(self.eeg_path_list_dir[0]))):
            aasm_rk_name = 0
        else:
            aasm_rk_name = 1
        eeg_path = os.path.join(self.eeg_path_list_dir[aasm_rk_name], self.eeg_path_lists[idx])
        ecg_path = os.path.join(self.ecg_path_list_dir[aasm_rk_name], self.ecg_path_lists[idx])
        slp_stg_path = os.path.join(self.slp_stg_path_list_dir[aasm_rk_name], self.slp_stg_path_lists[idx])
        # tens_eeg = torch.from_numpy(np.genfromtxt(eeg_path, delimiter=","))
        # tens_ecg = torch.from_numpy(np.genfromtxt(ecg_path, delimiter=","))
        # tens_slp_stg = torch.from_numpy(np.genfromtxt(slp_stg_path, delimiter=","))
        # eeg_file_data = torch.tensor(tens_eeg)
        # ecg_file_data = torch.tensor(tens_ecg)
        # slp_stg_file_data = torch.tensor(tens_slp_stg)
        # eeg_file_data = tens_eeg.clone().detach()
        # ecg_file_data = tens_ecg.clone().detach()
        # slp_stg_file_data = tens_slp_stg.clone().detach()
        # slp_stg_file_data = torch.tensor(tens_slp_stg)
        
        eeg_file_data = torch.tensor(np.genfromtxt(eeg_path, delimiter=","))
        ecg_file_data = torch.tensor(np.genfromtxt(ecg_path, delimiter=","))
        slp_stg_file_data = torch.tensor(np.genfromtxt(slp_stg_path, delimiter=","))      
        print(eeg_path)
        print("eeg_data_size", len(eeg_file_data))    
        return eeg_file_data, ecg_file_data, slp_stg_file_data

class MASSBatchSmallDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        self.n_workers = 32
        self.batch_size = 256
        self.eeg_train = self.eeg_val = self.eeg_test = torch.Tensor([])
        self.ecg_train = self.ecg_val = self.ecg_test = torch.Tensor([])
        self.slp_stg_train = self.slp_stg_val = self.slp_stg_test = torch.Tensor([])

        # SMALL DATASET #
        self.train_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/train/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/train/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/train/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/train/'])
        self.val_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/val/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/val/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/val/'])
        self.test_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/EEG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/EEG/test/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/ECG/test/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/AASM/Sleep_stages/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Small_Batchwise_Data/4_class/RK/Sleep_stages/test/'])
        
        # 50 BATCHES DATASET #
        # self.train_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/EEG/train/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/ECG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/ECG/train/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/Sleep_stages/train/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/Sleep_stages/train/'])
        # self.val_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/EEG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/EEG/val/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/ECG/val/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/Sleep_stages/val/'])
        # self.test_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/EEG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/EEG/test/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/ECG/test/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/AASM/Sleep_stages/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Fifty_Batchwise_Data/4_class/RK/Sleep_stages/test/'])
        # FULL DATASET #
        # self.train_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/train/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/train','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/train/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/train/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/train/'])
        # self.val_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/val/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/val','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/val/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/val/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/val/'])
        # self.test_dataset = MASSDataset(eeg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/EEG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/test/'], ecg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/ECG/test','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/test/'], slp_stg_path_lists=['/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/AASM/Sleep_stages/test/','/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/test/'])
       
        # for idx, tens in enumerate(self.train_dataset):
        #     pass 
        # #     # import pdb; pdb.set_trace()

        # self.train_class_weights = class_weight.compute_class_weight('balanced',np.arange(4),y.numpy())
        # self.train_class_weights = torch.tensor(self.train_class_weights,dtype=torch.float).cuda()

    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,num_workers=self.n_workers, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,num_workers=self.n_workers, pin_memory=True, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,num_workers=self.n_workers, batch_size=self.batch_size)


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

