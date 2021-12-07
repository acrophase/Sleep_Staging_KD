import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl 
import numpy as np
import glob
import random
import mne
from scipy import signal
from sklearn import preprocessing

# SS1_Channels = ['EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3', 'EEG F3-CLE', 'EEG F4-CLE', 'EEG C3-CLE', 'EEG C4-CLE', 'EEG O1-CLE', 'EEG O2-CLE', 'ECG ECGI', 'EMG Ant Tibial L', 'EMG Ant Tibial R', 'Resp Thermistor', 'Resp Cannula', 'Resp Belt Thor', 'Resp Belt Abdo', 'EEG F7-CLE', 'EEG F8-CLE', 'EEG T3-CLE', 'EEG T4-CLE', 'EEG T5-CLE', 'EEG T6-CLE', 'EEG P3-CLE', 'EEG P4-CLE', 'EEG Fz-CLE', 'EEG Cz-CLE', 'EEG Pz-CLE', 'EEG A2-CLE', 'SaO2 SaO2']
# SS2_Channels = ['EEG Fp1-CLE', 'EEG Fp2-CLE', 'EEG F3-CLE', 'EEG F4-CLE', 'EEG F7-CLE', 'EEG F8-CLE', 'EEG C3-CLE', 'EEG C4-CLE', 'EEG P3-CLE', 'EEG P4-CLE', 'EEG O1-CLE', 'EEG O2-CLE', 'EEG T3-CLE', 'EEG T4-CLE', 'EEG T5-CLE', 'EEG T6-CLE', 'EEG Fpz-CLE', 'EEG Cz-CLE', 'EEG Pz-CLE', 'EOG Upper Vertic', 'EOG Lower Vertic', 'EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin', 'ECG ECGI', 'Resp Nasal', 'EEG A2-CLE']
# SS3_Channels = ['EOG Left Horiz', 'EOG Right Horiz', 'EEG Fp1-LER', 'EEG Fp2-LER', 'EEG F7-LER', 'EEG F8-LER', 'EEG F3-LER', 'EEG F4-LER', 'EEG T3-LER', 'EEG T4-LER', 'EEG C3-LER', 'EEG C4-LER', 'EEG T5-LER', 'EEG T6-LER', 'EEG P3-LER', 'EEG P4-LER', 'EEG O1-LER', 'EEG O2-LER', 'EEG Fz-LER', 'EEG Cz-LER', 'EEG Pz-LER', 'EEG Oz-LER', 'EEG A2-LER', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3', 'ECG ECGI']
# SS4_Channels = ['EEG C3-CLE', 'EEG C4-CLE', 'EEG O1-CLE', 'EEG O2-CLE', 'EOG Upper Vertic', 'EOG Lower Vertic', 'EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin', 'ECG ECGI', 'ECG ECGII', 'ECG ECGIII', 'Resp Nasal', 'EEG A2-CLE', 'SaO2 SaO2']
# SS5_Channels = ['EOG Left Horiz', 'EOG Right Horiz', 'EEG Fp1-LER', 'EEG Fp2-LER', 'EEG F7-LER', 'EEG F8-LER', 'EEG F3-LER', 'EEG F4-LER', 'EEG T3-LER', 'EEG T4-LER', 'EEG C3-LER', 'EEG C4-LER', 'EEG T5-LER', 'EEG T6-LER', 'EEG P3-LER', 'EEG P4-LER', 'EEG O1-LER', 'EEG O2-LER', 'EEG Fz-LER', 'EEG Cz-LER', 'EEG Pz-LER', 'EEG Oz-LER', 'EEG A2-LER', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3', 'ECG ECGI']

# AASM:
# SS1_ch = ['EEG C3-CLE', 'EEG A2-CLE', 'ECG ECGI']
# SS3_ch = ['EEG C3-LER', 'EEG A2-LER', 'ECG ECGI']

# R_K:
# SS2_ch = ['EEG C3-CLE', 'EEG A2-CLE', 'ECG ECGI']
# SS4_ch = ['EEG C3-CLE', 'EEG A2-CLE', 'ECG ECGII']
# SS5_ch = ['EEG C3-LER', 'EEG A2-LER', 'ECG ECGI']



class MassDataset(Dataset):
    def  __init__(
        self,
        data_dir=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.MASS_files = glob.glob(os.path.join(self.data_dir,'*_EDF'))         #### MASS_dataset subset folders
        self.MASS_files.sort()
        # self.AASM = [self.MASS_files[0]] ### TAKING ONLY SS1 in consideration
        
        self.AASM = [self.MASS_files[0],self.MASS_files[2]]  ## TAKING SS1 and SS3
        self.RK = [self.MASS_files[1],self.MASS_files[3],self.MASS_files[4]]

        self.ecg_train2 = torch.Tensor([])
        self.eeg_trainc3a2 = torch.Tensor([])
        self.slp_stg_train = torch.Tensor([])
        
        self.ecg_val2 = torch.Tensor([])
        self.eeg_valc3a2 = torch.Tensor([])
        self.slp_stg_val = torch.Tensor([])

        self.ecg_test2 = torch.Tensor([])
        self.eeg_testc3a2 = torch.Tensor([])
        self.slp_stg_test = torch.Tensor([])
        
        self.n_slpstg=0

        
        for (outer_index,subset) in enumerate(self.AASM):
            # import pdb;pdb.set_trace()
            biosig_files = glob.glob(os.path.join(subset,'*PSG.edf'))    #### Biosignal files contain Polysomnography signals
            biosig_files.sort()

            ### Creating a 80-10-10 split ###
            ids = [i for i in range(len(biosig_files))]
            random.shuffle(ids)
            train_ids = ids[:int(len(biosig_files) * 0.8)]
            val_ids = ids[int(len(biosig_files) * 0.8):int(len(biosig_files) * 0.9)]
            test_ids = ids[int(len(biosig_files) * 0.9):]
            
            
            for (index,file) in enumerate(biosig_files):
                # import pdb;pdb.set_trace()
                ### ANNOTATION PROCESSING
                annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
                base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
                sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
                # slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}  ## 4 Class
                slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 1,"4" : 1,"R" : 2}  ## 3 Class
                slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
                sleep_epoch_len = 30  # in seconds
                
                ### BIOSIGNAL PROCESSING
                psg4 = mne.io.read_raw_edf(file)
                fs = int(psg4.info['sfreq'])

                ECG_ch = [i for i in psg4.ch_names if i.startswith('ECG ECG')]
                EEGC3_ch = [i for i in psg4.ch_names if i.startswith('EEG C3-')]
                EEGA2_ch = [i for i in psg4.ch_names if i.startswith('EEG A2')]

                ecg2= (-1)*psg4[ECG_ch[0]][0][0]    # Inverse of ideal waveform was available from dataset, so correcting it by * (-1)

                if EEGC3_ch == ['EEG C3-CLE']:
                    eegc3a2= psg4[EEGC3_ch[0]][0][0] - psg4[EEGA2_ch[0]][0][0]  # Getting C3 channel with A2 reference 

                elif EEGC3_ch == ['EEG C3-LER']:
                    eegc3a2= psg4[EEGC3_ch[0]][0][0]   # Getting C3 channel 
                
                # import pdb;pdb.set_trace()
                ##### ONSET OF ANNOTATION ####
                annot_onset = base4.onset[0] ## value in seconds
                #### Eliminating signal with no annotation
                ecg2 = ecg2[int(annot_onset)*fs::]
                eegc3a2 = eegc3a2[int(annot_onset)*fs::]
                
                ## Resampling 
                resamp_srate= 200
                num_windows = int(min(len(ecg2) / fs / sleep_epoch_len, len(base4)))
                
                min_max_scaler = preprocessing.MinMaxScaler()
                
                ## NORMALIZATION AND WINDOWING OF INPUT SIGNALS
                windowed_ecg2 = [signal.resample(min_max_scaler.fit_transform(ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
                windowed_eegc3a2 = [signal.resample(min_max_scaler.fit_transform(eegc3a2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
                
                if index in train_ids:

                    self.ecg_train2 = torch.cat([self.ecg_train2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_trainc3a2 = torch.cat([self.eeg_trainc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_train = torch.cat([self.slp_stg_train,torch.tensor(slp_stg_coarse[0:num_windows])])

                elif index in val_ids:

                    self.ecg_val2 = torch.cat([self.ecg_val2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_valc3a2 = torch.cat([self.eeg_valc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_val = torch.cat([self.slp_stg_val,torch.tensor(slp_stg_coarse[0:num_windows])])

                elif index in test_ids:

                    self.ecg_test2 = torch.cat([self.ecg_test2,torch.tensor(windowed_ecg2)], dim=0)
                    self.eeg_testc3a2 = torch.cat([self.eeg_testc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
                    self.slp_stg_test = torch.cat([self.slp_stg_test,torch.tensor(slp_stg_coarse[0:num_windows])])
                    
                print(self.ecg_train2.shape)
                print(self.ecg_val2.shape)
                print(self.ecg_test2.shape)
                self.n_slpstg += num_windows
            
            ### Removing -2 labels from train data
            self.ecg_train2 = self.ecg_train2[self.slp_stg_train != -2]
            self.eeg_trainc3a2 = self.eeg_trainc3a2[self.slp_stg_train != -2]
            self.slp_stg_train = self.slp_stg_train[self.slp_stg_train != -2]

            ### Removing -2 labels from val data
            self.ecg_val2 = self.ecg_val2[self.slp_stg_val != -2]
            self.eeg_valc3a2 = self.eeg_valc3a2[self.slp_stg_val != -2]
            self.slp_stg_val = self.slp_stg_val[self.slp_stg_val != -2]

            ### Removing -2 labels from test data
            self.ecg_test2 = self.ecg_test2[self.slp_stg_test != -2]
            self.eeg_testc3a2 = self.eeg_testc3a2[self.slp_stg_test != -2]
            self.slp_stg_test = self.slp_stg_test[self.slp_stg_test != -2]

        ## CHECKING DISTRIBUTION OF CLASSES ##
        _,count_train = self.slp_stg_train.unique(return_counts = True)
        _,count_val = self.slp_stg_val.unique(return_counts = True)
        _,count_test = self.slp_stg_test.unique(return_counts = True)

        self.distribution = torch.stack([count_train/sum(count_train), count_val/sum(count_val), count_test/sum(count_test)],dim=0)
        print(self.distribution)
        #### SAVING TRAIN-VAL-TEST DATA in PT_Files #####
        import pdb;pdb.set_trace()

        save_path = '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch_subjectwisesplit/3class/ALL_DATA/AASM/'

        self.train_data = torch.stack([self.eeg_trainc3a2,self.ecg_train2],dim=0)
        torch.save(self.train_data, save_path + 'eeg_ecg_1ch_train.pt')
        torch.save(self.slp_stg_train, save_path + 'slp_stg_train_lbl.pt')

        self.val_data = torch.stack([self.eeg_valc3a2,self.ecg_val2],dim=0)
        torch.save(self.val_data, save_path + 'eeg_ecg_1ch_eval.pt')
        torch.save(self.slp_stg_val, save_path + 'slp_stg_eval_lbl.pt')
        
        self.test_data = torch.stack([self.eeg_testc3a2,self.ecg_test2],dim=0)
        torch.save(self.test_data, save_path + 'eeg_ecg_1ch_test.pt')
        torch.save(self.slp_stg_test, save_path + 'slp_stg_test_lbl.pt')
        
        
    def __getitem__(self,index):
#         # dataset indexing
        # import pdb;pdb.set_trace()
        return self.train_data[index],self.slp_stg_train[index],self.val_data[index],self.slp_stg_val[index],self.test_data[index],self.slp_stg_test[index]
        # self.ecg_train[index],
        # self.eeg_train[index],self.slp_stg_train[index],self.ecg_test[index],self.eeg_test[index],self.slp_stg_test[index]
        
        
    def __len__(self):
        return self.n_slpstg


data_path = "/media/Sentinel_2/Dataset/Vaibhav/MASS/MASS_BIOSIG/"
                                                                                     
if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)
    
    MassDataset(data_dir=data_path)