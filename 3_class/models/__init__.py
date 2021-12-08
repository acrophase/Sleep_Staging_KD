from .ecg_base import ECG_BASE_Model
from .eeg_base import EEG_BASE_Model
from .KD_TEMP import SD_CL_model
from .FEAT_TRAINING import FEAT_TRAINING_model
from .FEAT_WCE import AT_CL_model
from .FEAT_TEMP import AT_SD_CL_model

available_models = {
                    "ecg_base": ECG_BASE_Model,
                    "eeg_base": EEG_BASE_Model,
                    "kd_temp": SD_CL_model,
                    "feat_train": FEAT_TRAINING_model,
                    "feat_wce": AT_CL_model,
                    "feat_temp": AT_SD_CL_model,
                    }