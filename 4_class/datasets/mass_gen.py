# 4 CLASS ALL DATA AASM & RK
# THIS CODE INVERTS ECG SIGNALS WHEN REQUIRED, APPLIES 60 HZ FILTERING,
# CONTAINS THE GENERATOR FUNCTION APPLIED TO EEG, ECG, SPL_STG COMBINED
# AND LOADS TO CUSTOM DATALOADER
import random
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
import os
import glob
import random
import mne
from scipy import signal
from sklearn import preprocessing
from sklearn.utils import class_weight
import pytorch_lightning as pl

class MassDataModuleGen(pl.LightningDataModule):
    def  __init__(
        self,
        batch_size=256,
        n_workers=32,
        *args,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.data_dir = "/media/Sentinel_2/Dataset/Vaibhav/MASS/MASS_BIOSIG/"
        self.MASS_files = glob.glob(os.path.join(self.data_dir,'*_EDF'))      #### MASS_dataset subset folders
        self.MASS_files.sort()
        self.AASM = [self.MASS_files[0],self.MASS_files[2]]  ## TAKING SS1 and SS3  
        self.RK = [self.MASS_files[1],self.MASS_files[3],self.MASS_files[4]] ## TAKING SS2,SS4 and SS5
        
        ## Initializing all tensors ##
        self.ecg_aasm_train2 = torch.Tensor([])
        self.eeg_aasm_trainc3a2 = torch.Tensor([])
        self.slp_stg_aasm_train = torch.Tensor([])
        
        self.ecg_aasm_val2 = torch.Tensor([])
        self.eeg_aasm_valc3a2 = torch.Tensor([])
        self.slp_stg_aasm_val = torch.Tensor([])

        self.ecg_aasm_test2 = torch.Tensor([])
        self.eeg_aasm_testc3a2 = torch.Tensor([])
        self.slp_stg_aasm_test = torch.Tensor([])

        self.ecg_rk_train2 = torch.Tensor([])
        self.eeg_rk_trainc3a2 = torch.Tensor([])
        self.slp_stg_rk_train = torch.Tensor([])
        
        self.ecg_rk_val2 = torch.Tensor([])
        self.eeg_rk_valc3a2 = torch.Tensor([])
        self.slp_stg_rk_val = torch.Tensor([])

        self.ecg_rk_test2 = torch.Tensor([])
        self.eeg_rk_testc3a2 = torch.Tensor([])
        self.slp_stg_rk_test = torch.Tensor([])

        self.ecg_train2 = torch.Tensor([])
        self.eeg_trainc3a2 = torch.Tensor([])
        self.slp_stg_train = torch.Tensor([])
        
        self.ecg_val2 = torch.Tensor([])
        self.eeg_valc3a2 = torch.Tensor([])
        self.slp_stg_val = torch.Tensor([])

        self.ecg_test2 = torch.Tensor([])
        self.eeg_testc3a2 = torch.Tensor([])
        self.slp_stg_test = torch.Tensor([])
        
        self.n_aasm_slpstg=0
        self.n_rk_slpstg=0

        ##### AASM PROCESSING BLOCK #####
        count = 0
        for (outer_index,subset) in enumerate(self.AASM):
            biosig_files = glob.glob(os.path.join(subset,'*PSG.edf'))    #### Biosignal files contain Polysomnography signals
            biosig_files.sort()
            ### Creating a 80-10-10 split ###
            ids = [i for i in range(len(biosig_files))]
            random.shuffle(ids)
            train_ids = ids[:int(len(biosig_files) * 0.8)]
            val_ids = ids[int(len(biosig_files) * 0.8):int(len(biosig_files) * 0.9)]
            test_ids = ids[int(len(biosig_files) * 0.9):]
            
            
            for (index,file) in enumerate(biosig_files):
                
                ### ANNOTATION PROCESSING
                annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
                base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
                sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
                slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}
                slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
                sleep_epoch_len = 30  # in seconds
                
                ### BIOSIGNAL PROCESSING
                psg4 = mne.io.read_raw_edf(file)
                fs = int(psg4.info['sfreq'])
                
                ECG_ch = ['ECG ECGI']
                EEGC3_ch = [i for i in psg4.ch_names if i.startswith('EEG C3-')]
                EEGA2_ch = [i for i in psg4.ch_names if i.startswith('EEG A2')]

                ### INVERSION BASED ON GROUND TRUTH
                aasm_inv_gt = pd.read_csv("/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/SS_KD/Dataset_split/Invert_Ground_Truth_AASM.csv")
                gt = aasm_inv_gt['Ground Truth']                
                if gt[count] == 'INV':
                    ecg2= (-1)*psg4[ECG_ch[0]][0][0] # Inverse of ideal waveform was available from dataset, so correcting it by * (-1)
                else:
                    ecg2= psg4[ECG_ch[0]][0][0]
                count+=1

                ### FILTERING OF 60 Hz
                b, a = signal.iirnotch(60.0, 30.0, fs) #60 Hz to be filtered, at Quality factor = 30
                outputSignal = signal.filtfilt(b, a, ecg2)
                ecg2 = outputSignal

                if EEGC3_ch == ['EEG C3-CLE']:
                    eegc3a2= psg4[EEGC3_ch[0]][0][0] - psg4[EEGA2_ch[0]][0][0]  # Getting C3 channel with A2 reference 

                elif EEGC3_ch == ['EEG C3-LER']:
                    eegc3a2= psg4[EEGC3_ch[0]][0][0]   # Getting C3 channel 
                
                ##### ONSET OF ANNOTATION ####
                annot_onset = base4.onset[0] ## value in seconds
                #### Eliminating signal with no annotation
                ecg2 = ecg2[int(annot_onset)*fs::]
                eegc3a2 = eegc3a2[int(annot_onset)*fs::]

                ## Resampling 
                resamp_srate= 200
                #num_windows = int(min(len(ecg2) / fs / sleep_epoch_len, len(base4)))
                num_windows = int(len(base4))
                min_max_scaler = preprocessing.MinMaxScaler()
                
                ### NORMALIZATION AND WINDOWING OF INPUT SIGNALS
                windowed_ecg2 = [signal.resample(min_max_scaler.fit_transform(ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
                windowed_eegc3a2 = [signal.resample(min_max_scaler.fit_transform(eegc3a2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]

                if index in train_ids:

                    self.ecg_aasm_train2 = torch.cat([self.ecg_aasm_train2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_aasm_trainc3a2 = torch.cat([self.eeg_aasm_trainc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_aasm_train = torch.cat([self.slp_stg_aasm_train,torch.tensor(slp_stg_coarse[0:num_windows])])

                elif index in val_ids:

                    self.ecg_aasm_val2 = torch.cat([self.ecg_aasm_val2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_aasm_valc3a2 = torch.cat([self.eeg_aasm_valc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_aasm_val = torch.cat([self.slp_stg_aasm_val,torch.tensor(slp_stg_coarse[0:num_windows])])

                elif index in test_ids:

                    self.ecg_aasm_test2 = torch.cat([self.ecg_aasm_test2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_aasm_testc3a2 = torch.cat([self.eeg_aasm_testc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_aasm_test = torch.cat([self.slp_stg_aasm_test,torch.tensor(slp_stg_coarse[0:num_windows])])

                print(self.ecg_aasm_train2.shape)
                print(self.ecg_aasm_val2.shape)
                print(self.ecg_aasm_test2.shape)
                self.n_aasm_slpstg += num_windows

            ### Removing -2 labels from train data
            self.ecg_aasm_train2 = self.ecg_aasm_train2[self.slp_stg_aasm_train != -2]
            self.eeg_aasm_trainc3a2 = self.eeg_aasm_trainc3a2[self.slp_stg_aasm_train != -2]
            self.slp_stg_aasm_train = self.slp_stg_aasm_train[self.slp_stg_aasm_train != -2]

            ### Removing -2 labels from val data
            self.ecg_aasm_val2 = self.ecg_aasm_val2[self.slp_stg_aasm_val != -2]
            self.eeg_aasm_valc3a2 = self.eeg_aasm_valc3a2[self.slp_stg_aasm_val != -2]
            self.slp_stg_aasm_val = self.slp_stg_aasm_val[self.slp_stg_aasm_val != -2]

            ### Removing -2 labels from test data
            self.ecg_aasm_test2 = self.ecg_aasm_test2[self.slp_stg_aasm_test != -2]
            self.eeg_aasm_testc3a2 = self.eeg_aasm_testc3a2[self.slp_stg_aasm_test != -2]
            self.slp_stg_aasm_test = self.slp_stg_aasm_test[self.slp_stg_aasm_test != -2]

        ##### RK PROCESSING BLOCK #####
        count = 0
        for (outer_index,subset) in enumerate(self.RK):
            biosig_files = glob.glob(os.path.join(subset,'*PSG.edf'))    #### Biosignal files contain Polysomnography signals
            biosig_files.sort()

            ### Creating a 80-10-10 split ###
            ids = [i for i in range(len(biosig_files))]
            random.shuffle(ids)
            train_ids = ids[:int(len(biosig_files) * 0.8)]
            val_ids = ids[int(len(biosig_files) * 0.8):int(len(biosig_files) * 0.9)]
            test_ids = ids[int(len(biosig_files) * 0.9):]
            
            
            for (index,file) in enumerate(biosig_files):
                ### ANNOTATION PROCESSING
                annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
                base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
                sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
                slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}
                slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
                sleep_epoch_len = 20  # in seconds
                
                ### BIOSIGNAL PROCESSING
                psg4 = mne.io.read_raw_edf(file)
                fs = int(psg4.info['sfreq'])

                EEGC_ch = [i for i in psg4.ch_names if i.startswith('EEG C3-')]
                EEGA_ch = [i for i in psg4.ch_names if i.startswith('EEG A')]

                ### EEG SIGNAL EXTRACTION FROM SUITABLE CHANNELS
                if EEGA_ch == ['EEG A1-CLE']:
                    EEGC_ch = ['EEG C4-CLE']
                    eegc3a2= psg4[EEGC_ch[0]][0][0] - psg4[EEGA_ch[0]][0][0]  # Getting C4-A1 where C3-A2 not available (IN SS2)
                elif EEGA_ch == ['EEG A2-CLE']:
                    EEGC_ch = ['EEG C3-CLE']
                    eegc3a2= psg4[EEGC_ch[0]][0][0] - psg4[EEGA_ch[0]][0][0]  # Getting C3 channel with A2 reference
                elif EEGC_ch == ['EEG C3-LER']:
                    eegc3a2= psg4[EEGC_ch[0]][0][0]   # Getting C3 channel (IN SS5)

                ### ECG SIGNAL EXTRACTION FROM SUITABLE CHANNELS
                if outer_index == 1:  ## FOR SS4 ECG lead 2 to be used as available
                    ECG_ch = ['ECG ECGII']
                else:
                    ECG_ch = [i for i in psg4.ch_names if i.startswith('ECG ECG')]
                # ECG_ch = ['ECG ECGI']

                ### INVERSION BASED ON GROUND TRUTH
                rk_inv_gt = pd.read_csv("/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/SS_KD/Dataset_split/Invert_Ground_Truth_RK.csv")
                gt = rk_inv_gt['Ground Truth']
                if gt[count] == 'INV':
                    ecg2= (-1)*psg4[ECG_ch[0]][0][0]    # Inverse of ideal waveform was available from dataset, so correcting it by * (-1)
                else:
                    ecg2= psg4[ECG_ch[0]][0][0]
                count+=1

                ### FILTERING OF 60 Hz 
                b, a = signal.iirnotch(60.0, 30.0, fs) #60 Hz to be filtered, at Quality factor = 30
                outputSignal = signal.filtfilt(b, a, ecg2)
                ecg2 = outputSignal

                #### ONSET OF ANNOTATION ####
                annot_onset = base4.onset[0] ## value in seconds
                #### Eliminating signal with no annotation
                ecg2 = ecg2[int(annot_onset)*fs::]
                eegc3a2 = eegc3a2[int(annot_onset)*fs::]

                ## Resampling 
                resamp_srate= 200
                #num_windows = int(min(len(ecg2) / fs / sleep_epoch_len, len(base4)))
                num_windows = int(len(base4))
                min_max_scaler = preprocessing.MinMaxScaler()
                
                ### NORMALIZATION AND WINDOWING OF INPUT SIGNALS
                new_sleep_epoch_len = 30
                windowed_ecg2_30s = [signal.resample(min_max_scaler.fit_transform(ecg2[((i+1) * fs * sleep_epoch_len) - (fs*5) : ((i+2) * fs * sleep_epoch_len)+(fs*5)].reshape(-1,1)).squeeze(1), resamp_srate*new_sleep_epoch_len) for i in range(num_windows-2)]# if fs != resamp_srate]
                windowed_eegc3a2_30s = [signal.resample(min_max_scaler.fit_transform(eegc3a2[((i+1) * fs * sleep_epoch_len)- (fs*5) : ((i+2)* fs * sleep_epoch_len) + (fs*5)].reshape(-1,1)).squeeze(1), resamp_srate*new_sleep_epoch_len) for i in range(num_windows-2)]# if fs != resamp_srate]

                if index in train_ids:

                    self.ecg_rk_train2 = torch.cat([self.ecg_rk_train2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_rk_trainc3a2 = torch.cat([self.eeg_rk_trainc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_rk_train = torch.cat([self.slp_stg_rk_train,torch.tensor(slp_stg_coarse[1:num_windows-1])])

                elif index in val_ids:

                    self.ecg_rk_val2 = torch.cat([self.ecg_rk_val2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_rk_valc3a2 = torch.cat([self.eeg_rk_valc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_rk_val = torch.cat([self.slp_stg_rk_val,torch.tensor(slp_stg_coarse[1:num_windows-1])])

                elif index in test_ids:

                    self.ecg_rk_test2 = torch.cat([self.ecg_rk_test2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_rk_testc3a2 = torch.cat([self.eeg_rk_testc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_rk_test = torch.cat([self.slp_stg_rk_test,torch.tensor(slp_stg_coarse[1:num_windows-1])])

                print(self.ecg_rk_train2.shape)
                print(self.ecg_rk_val2.shape)
                print(self.ecg_rk_test2.shape)
                self.n_rk_slpstg += num_windows
            
            ### Removing -2 labels from train data
            self.ecg_rk_train2 = self.ecg_rk_train2[self.slp_stg_rk_train != -2]
            self.eeg_rk_trainc3a2 = self.eeg_rk_trainc3a2[self.slp_stg_rk_train != -2]
            self.slp_stg_rk_train = self.slp_stg_rk_train[self.slp_stg_rk_train != -2]

            ### Removing -2 labels from val data
            self.ecg_rk_val2 = self.ecg_rk_val2[self.slp_stg_rk_val != -2]
            self.eeg_rk_valc3a2 = self.eeg_rk_valc3a2[self.slp_stg_rk_val != -2]
            self.slp_stg_rk_val = self.slp_stg_rk_val[self.slp_stg_rk_val != -2]

            ### Removing -2 labels from test data
            self.ecg_rk_test2 = self.ecg_rk_test2[self.slp_stg_rk_test != -2]
            self.eeg_rk_testc3a2 = self.eeg_rk_testc3a2[self.slp_stg_rk_test != -2]
            self.slp_stg_rk_test = self.slp_stg_rk_test[self.slp_stg_rk_test != -2]

        ## CHECKING DISTRIBUTION OF CLASSES ##
        # _,count_train = self.slp_stg_train.unique(return_counts = True)
        # _,count_val = self.slp_stg_val.unique(return_counts = True)
        # _,count_test = self.slp_stg_test.unique(return_counts = True)

        # self.distribution = torch.stack([count_train/sum(count_train), count_val/sum(count_val), count_test/sum(count_test)],dim=0)
        # print(self.distribution)

    ##### SPLITTING INTO BATCHES USING GENERATOR FUNCTION ######
    # Batch of AASM & RK generated individually, then concatenated and loaded to data loader
    def setup(self, stage = None):
        if stage == "fit": #training and validation data 
            iter_fit_aasm = 0
            iter_fit_rk = 0

            if iter_fit_aasm < len(self.AASM):
                self.eeg_aasm_train_batch = next(self.data_generator(self.eeg_aasm_trainc3a2))
                self.ecg_aasm_train_batch = next(self.data_generator(self.ecg_aasm_train2))
                self.slp_stg_aasm_train_batch = next(self.data_generator(self.slp_stg_aasm_train))

                self.eeg_aasm_val_batch = next(self.data_generator(self.eeg_aasm_valc3a2))
                self.ecg_aasm_val_batch = next(self.data_generator(self.ecg_aasm_val2))
                self.slp_stg_aasm_val_batch = next(self.data_generator(self.slp_stg_aasm_val))
                
                iter_fit_aasm+=1

            if iter_fit_rk < len(self.RK):
                self.eeg_rk_train_batch = next(self.data_generator(self.eeg_rk_trainc3a2))
                self.ecg_rk_train_batch = next(self.data_generator(self.ecg_rk_train2))
                self.slp_stg_rk_train_batch = next(self.data_generator(self.slp_stg_rk_train))

                self.eeg_rk_val_batch = next(self.data_generator(self.eeg_rk_valc3a2))
                self.ecg_rk_val_batch = next(self.data_generator(self.ecg_rk_val2))
                self.slp_stg_rk_val_batch = next(self.data_generator(self.slp_stg_rk_val))

                iter_fit_rk+=1

            self.eeg_train_batch = torch.cat([self.eeg_aasm_train_batch,self.eeg_rk_train_batch],dim=0)
            self.ecg_train_batch = torch.cat([self.ecg_aasm_train_batch,self.ecg_rk_train_batch],dim=0)
            self.slp_stg_train_batch = torch.cat([self.slp_stg_aasm_train_batch,self.slp_stg_rk_train_batch],dim=0)

            self.eeg_val_batch = torch.cat([self.eeg_aasm_val_batch,self.eeg_rk_val_batch],dim=0)
            self.ecg_val_batch = torch.cat([self.ecg_aasm_val_batch,self.ecg_rk_val_batch],dim=0)
            self.slp_stg_val_batch = torch.cat([self.slp_stg_aasm_val_batch,self.slp_stg_rk_val_batch],dim=0)
            
            self.train_data_batch = TensorDataset(self.eeg_train_batch, self.ecg_train_batch, self.slp_stg_train_batch)
            self.val_data_batch =   TensorDataset(self.eeg_val_batch,   self.ecg_val_batch,   self.slp_stg_val_batch)
                
                
            
        elif stage == "test": #test data
            iter_test_aasm = 0
            iter_test_rk = 0
         
            if iter_test_aasm < len(self.AASM):
                self.eeg_aasm_test_batch = next(self.data_generator(self.eeg_aasm_testc3a2))
                self.ecg_aasm_test_batch = next(self.data_generator(self.ecg_aasm_test2))
                self.slp_stg_aasm_test_batch = next(self.data_generator(self.slp_stg_aasm_test))

                iter_test_aasm+=1

            if iter_test_rk < len(self.RK):    
                self.eeg_rk_test_batch = next(self.data_generator(self.eeg_rk_testc3a2))
                self.ecg_rk_test_batch = next(self.data_generator(self.ecg_rk_test2))
                self.slp_stg_rk_test_batch = next(self.data_generator(self.slp_stg_rk_test))            
                
                iter_test_rk+=1

            self.eeg_test_batch = torch.cat([self.eeg_aasm_test_batch,self.eeg_rk_test_batch],dim=0)
            self.ecg_test_batch = torch.cat([self.ecg_aasm_test_batch,self.ecg_rk_test_batch],dim=0)
            self.slp_stg_test_batch = torch.cat([self.slp_stg_aasm_test_batch,self.slp_stg_rk_test_batch],dim=0)

            self.test_data_batch = TensorDataset(self.eeg_test_batch, self.ecg_test_batch,  self.slp_stg_test_batch)
                

    ##### GENERATOR FUNCTION #####   
    def data_generator(self, input_data):
        no_of_batches = int(np.floor(len(input_data)/self.batch_size))
        print("batches = ", no_of_batches)
        for index in range(no_of_batches):
            #generate 1 batch of data
            data_batch = input_data[index*self.batch_size : (index+100)*self.batch_size]
            yield data_batch      

    ##### LOADING DATA IN DATA LOADER #####
    def train_dataloader(self):
        """Return training dataloader."""
        # self.setup("fit")
        return DataLoader(
            self.train_data_batch,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )
    def val_dataloader(self):
        """Return validation dataloader."""
        # self.setup("fit")
        return DataLoader(
            self.val_data_batch,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
        )
    def test_dataloader(self):
        """Return test dataloader."""
        # self.setup("test")
        return DataLoader(
            self.test_data_batch,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )
    
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
    np.random.seed(42)
    random.seed(42)
    dm = MassDataModuleGen(batch_size=256,n_workers=32)
    dm.setup("fit")
    dm.setup("test")
