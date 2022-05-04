import os
from pyrsistent import v
import torch
from torch.utils.data import Dataset
import glob
import random
import mne
from sklearn.utils import class_weight
import numpy as np
from pytorch_lightning import seed_everything

class Class_Weights(Dataset):
    def  __init__(
        self,
        data_dir="/media/Sentinel_2/Dataset/Vaibhav/MASS/MASS_BIOSIG/",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.MASS_files = glob.glob(os.path.join(self.data_dir,'*_EDF'))         #### MASS_dataset subset folders
        self.MASS_files.sort()
        self.AASM = [self.MASS_files[0],self.MASS_files[2]]  ## TAKING SS1 and SS3
        self.RK = [self.MASS_files[1],self.MASS_files[3],self.MASS_files[4]] ## TAKING SS2,SS4 and SS5

        self.aasm_slp_stg_train = torch.Tensor([])
        self.rk_slp_stg_train = torch.Tensor([])
        self.aasm_slp_stg_val = torch.Tensor([])
        self.rk_slp_stg_val = torch.Tensor([])
        self.aasm_slp_stg_test = torch.Tensor([])
        self.rk_slp_stg_test = torch.Tensor([])

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
                annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
                base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
                sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
                slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}
                slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
                num_windows = int(len(base4))

                if index in train_ids:
                    self.aasm_slp_stg_train = torch.cat([self.aasm_slp_stg_train,torch.tensor(slp_stg_coarse[0:num_windows])])

                ### Removing -2 labels from data
                self.aasm_slp_stg_train = self.aasm_slp_stg_train[self.aasm_slp_stg_train != -2]


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
                annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
                base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
                sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
                slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}
                slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
                num_windows = int(len(base4))

                if index in train_ids:
                    self.rk_slp_stg_train = torch.cat([self.rk_slp_stg_train,torch.tensor(slp_stg_coarse[0:num_windows])])
                
                ### Removing -2 labels from data
                self.rk_slp_stg_train = self.rk_slp_stg_train[self.rk_slp_stg_train != -2]
        self.slp_stg_train = torch.cat([self.aasm_slp_stg_train,self.rk_slp_stg_train], dim = 0)
        y=self.slp_stg_train.cpu()
        self.train_class_weights = class_weight.compute_class_weight('balanced',np.arange(4),y.numpy().ravel())
        filename = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/MeMeA_Slp_staging_repo/Sleep_Staging_KD/4_class/datasets/class_weights.txt'
        np.savetxt(filename, self.train_class_weights)
      
if __name__ == "__main__":
    seed_everything(0, workers=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    Class_Weights()