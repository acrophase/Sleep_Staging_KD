import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl 
import numpy as np
import pandas as pd
import glob
import random
import mne
from scipy import signal
from sklearn import preprocessing
import argparse

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
        save_path=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.save_path = save_path
        self.MASS_files = glob.glob(os.path.join(self.data_dir,'*_EDF'))         #### MASS_dataset subset folders
        self.MASS_files.sort()
        self.AASM = [self.MASS_files[0],self.MASS_files[2]]  ## TAKING SS1 and SS3
        self.RK = [self.MASS_files[1],self.MASS_files[3],self.MASS_files[4]] ## TAKING SS2,SS4 and SS5

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
        self.batch_size = 256
        
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

                ## EEG SIGNAL EXTRACTION FROM SUITABLE CHANNELS
                if EEGA_ch == ['EEG A1-CLE']:
                    EEGC_ch = ['EEG C4-CLE']
                    eegc3a2= psg4[EEGC_ch[0]][0][0] - psg4[EEGA_ch[0]][0][0]  # Getting C4-A1 where C3-A2 not available (IN SS2)
                elif EEGA_ch == ['EEG A2-CLE']:
                    EEGC_ch = ['EEG C3-CLE']
                    eegc3a2= psg4[EEGC_ch[0]][0][0] - psg4[EEGA_ch[0]][0][0]  # Getting C3 channel with A2 reference
                elif EEGC_ch == ['EEG C3-LER']:
                    eegc3a2= psg4[EEGC_ch[0]][0][0]   # Getting C3 channel (IN SS5)

                ## ECG SIGNAL EXTRACTION FROM SUITABLE CHANNELS
                if outer_index == 1:  ## FOR SS4 ECG lead 2 to be used as available
                    ECG_ch = ['ECG ECGII']
                else:
                    ECG_ch = [i for i in psg4.ch_names if i.startswith('ECG ECG')]

                ## CHECKING IF SIGNAL IS INVERTED FROM GROUND TRUTH
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
                
                ## NORMALIZATION AND WINDOWING OF INPUT SIGNALS
                new_sleep_epoch_len = 30
                windowed_ecg2_30s = [signal.resample(min_max_scaler.fit_transform(ecg2[((i+1) * fs * sleep_epoch_len) - (fs*5) : ((i+2) * fs * sleep_epoch_len)+(fs*5)].reshape(-1,1)).squeeze(1), resamp_srate*new_sleep_epoch_len) for i in range(num_windows-2)]# if fs != resamp_srate]
                windowed_eegc3a2_30s = [signal.resample(min_max_scaler.fit_transform(eegc3a2[((i+1) * fs * sleep_epoch_len)- (fs*5) : ((i+2)* fs * sleep_epoch_len) + (fs*5)].reshape(-1,1)).squeeze(1), resamp_srate*new_sleep_epoch_len) for i in range(num_windows-2)]# if fs != resamp_srate]

                if index in train_ids:

                    self.ecg_train2 = torch.cat([self.ecg_train2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_trainc3a2 = torch.cat([self.eeg_trainc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_train = torch.cat([self.slp_stg_train,torch.tensor(slp_stg_coarse[1:num_windows-1])])

                elif index in val_ids:

                    self.ecg_val2 = torch.cat([self.ecg_val2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_valc3a2 = torch.cat([self.eeg_valc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_val = torch.cat([self.slp_stg_val,torch.tensor(slp_stg_coarse[1:num_windows-1])])

                elif index in test_ids:

                    self.ecg_test2 = torch.cat([self.ecg_test2,torch.tensor(windowed_ecg2_30s)], dim=0)
                    self.eeg_testc3a2 = torch.cat([self.eeg_testc3a2,torch.tensor(windowed_eegc3a2_30s)], dim=0)
                    self.slp_stg_test = torch.cat([self.slp_stg_test,torch.tensor(slp_stg_coarse[1:num_windows-1])])

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

        # ## CHECKING DISTRIBUTION OF CLASSES ##
        # _,count_train = self.slp_stg_train.unique(return_counts = True)
        # _,count_val = self.slp_stg_val.unique(return_counts = True)
        # _,count_test = self.slp_stg_test.unique(return_counts = True)

        # self.distribution = torch.stack([count_train/sum(count_train), count_val/sum(count_val), count_test/sum(count_test)],dim=0)
        # print(self.distribution)

        #### SAVING TRAIN-VAL-TEST DATA in PT_Files #####
        train_eeg_data_gen = self.data_generator(self.eeg_trainc3a2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(len(self.eeg_trainc3a2)-self.batch_size*50):
            batch_eeg_train = next(train_eeg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_eeg_train.numpy())
            array_batch.to_csv(save_path + "train/batch" + str(file_iter+6) + "/eeg_train_batch_"+ str(file_iter+6) +".csv", index = False, header = False)
            # torch.save(batch_eeg_train, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/eeg_train_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        train_ecg_data_gen = self.data_generator(self.ecg_train2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(len(self.ecg_train2)-self.batch_size*50):
            batch_ecg_train = next(train_ecg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_ecg_train.numpy())
            array_batch.to_csv(save_path + "train/batch" + str(file_iter+6) + "/ecg_train_batch_"+ str(file_iter+6) +".csv", index = False, header = False)
            # torch.save(batch_ecg_train, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/ecg_train_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        train_slp_stg_data_gen = self.data_generator(self.slp_stg_train)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(len(self.slp_stg_train)-self.batch_size*50):
            batch_slp_stg_train = next(train_slp_stg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_slp_stg_train.numpy())
            array_batch.to_csv(save_path + "train/batch" + str(file_iter+6) + "/slp_stg_train_batch_"+ str(file_iter+6) +".csv", index = False, header = False)
            # torch.save(batch_slp_stg_train, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_Stages/slp_stg_train_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        val_eeg_data_gen = self.data_generator(self.eeg_valc3a2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.eeg_valc3a2)-self.batch_size*50)):
            batch_eeg_val = next(val_eeg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_eeg_val.numpy())
            array_batch.to_csv(save_path + "val/batch" + str(file_iter+1) + "/eeg_val_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_eeg_val, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/eeg_val_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        val_ecg_data_gen = self.data_generator(self.ecg_val2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.ecg_val2)-self.batch_size*50)):
            batch_ecg_val = next(val_ecg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_ecg_val.numpy())
            array_batch.to_csv(save_path + "val/batch" + str(file_iter) + "/ecg_val_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_ecg_val, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/ecg_val_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        val_slp_stg_data_gen = self.data_generator(self.slp_stg_val)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.slp_stg_val)-self.batch_size*50)):
            batch_slp_stg_val = next(val_slp_stg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_slp_stg_val.numpy())
            array_batch.to_csv(save_path + "val/batch" + str(file_iter) + "/slp_stg_val_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_slp_stg_val, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/slp_stg_val_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        test_eeg_data_gen = self.data_generator(self.eeg_testc3a2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.eeg_testc3a2)-self.batch_size*50)):
            batch_eeg_test = next(test_eeg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_eeg_test.numpy())
            array_batch.to_csv(save_path + "test/batch" + str(file_iter) + "/eeg_test_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_eeg_test, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/EEG/eeg_test_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        test_ecg_data_gen = self.data_generator(self.ecg_test2)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.ecg_test2)-self.batch_size*50)):
            batch_ecg_test = next(test_ecg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_ecg_test.numpy())
            array_batch.to_csv(save_path + "test/batch" + str(file_iter) + "/ecg_test_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_ecg_test, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/ECG/ecg_test_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
        
        test_slp_stg_data_gen = self.data_generator(self.slp_stg_test)
        loop_iter = 0
        file_iter = 0
        while loop_iter in range(abs(len(self.slp_stg_test)-self.batch_size*50)):
            batch_slp_stg_test = next(test_slp_stg_data_gen)
            file_iter+=1
            array_batch = pd.DataFrame(batch_slp_stg_test.numpy())
            array_batch.to_csv(save_path + "test/batch" + str(file_iter) + "/lp_stg_test_batch_"+ str(file_iter+1) +".csv", index = False, header = False)
            # torch.save(batch_slp_stg_test, '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Batchwise_Data/4_class/RK/Sleep_stages/slp_stg_test_batch' + str(file_iter+1) + '.pt')
            loop_iter+=self.batch_size*50
            
    def data_generator(self, input_data):
        no_of_batches = int(np.floor(len(input_data)/(self.batch_size*50)))
        print("batches = ", no_of_batches)
        if no_of_batches == 0:
                yield input_data
        else:
            gen_index = 0
            for gen_index in range(no_of_batches):
                #generate 1 batch of data
                data_batch = input_data[gen_index*self.batch_size*50 : (gen_index+1)*self.batch_size*50] 
                #if no. of batches to be generated is changed from 100, reflect change in updation of self.setup_iter_tesk_aasm/rk
                yield data_batch 
                    
    def __getitem__(self,index):
#         # dataset indexing
        return self.train_data[index],self.slp_stg_train[index],self.val_data[index],self.slp_stg_val[index],self.test_data[index],self.slp_stg_test[index]
        # self.ecg_train[index],
        # self.eeg_train[index],self.slp_stg_train[index],self.ecg_test[index],self.eeg_test[index],self.slp_stg_test[index]
        
        
    def __len__(self):
        return self.n_slpstg


def argparsing():
    parser = argparse.ArgumentParser(description='RUN Dataset split')
    # parser.add_argument("--data_path", default="/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch_subjectwisesplit/ALL_DATA/AASM/", help= 'Enter path to data PT files')
    parser.add_argument("--data_path", default="/media/Sentinel_2/Dataset/Vaibhav/MASS/MASS_BIOSIG/", help= 'Enter path to data')
    #parser.add_argument("--save_path", default='/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/POCT/4_class/R_K_30s/', help= 'Enter path to save files')
    parser.add_argument("--save_path", help= 'Enter path to save files')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    args= argparsing()
    data_path= args.data_path
    save_path = args.save_path
    MassDataset(data_dir=data_path, save_path='/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/Nikhil_Datasplit/4_class/')