from .ecg_base import ECG_BASE_Model
from .eeg_base import EEG_BASE_Model
from .KD_TEMP import SD_CL_model
from .FEAT_WCE import AT_CL_model
from .FEAT_TEMP import AT_SD_CL_model
from .FEAT_TRAINING import FEAT_TRAINING_model

from .ecg_base_dice import ECG_BASE_DICE_Model
from .eeg_base_dice import EEG_BASE_DICE_Model
from .KD_TEMP_DICE import SD_CL_DICE_model
from .FEAT_TRAINING_DICE import FEAT_TRAINING_DICE_model
from .FEAT_DICE import AT_CL_DICE_model
from .FEAT_TEMP_DICE import AT_SD_CL_DICE_model


available_models = {"ecg_base": ECG_BASE_Model, 
                    "eeg_base": EEG_BASE_Model, 
                    "kd_temp": SD_CL_model, 
                    "feat_train": FEAT_TRAINING_model,
                    "feat_wce": AT_CL_model, 
                    "feat_temp": AT_SD_CL_model,

                    "ecg_base_dice": ECG_BASE_DICE_Model,
                    "eeg_base_dice": EEG_BASE_DICE_Model,
                    "kd_temp_dice": SD_CL_DICE_model,
                    "feat_train_dice": FEAT_TRAINING_DICE_model,
                    "feat_dice": AT_CL_DICE_model,
                    "feat_temp_dice": AT_SD_CL_DICE_model,
                    }