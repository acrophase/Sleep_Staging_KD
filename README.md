# Sleep_Staging_Knowledge Distillation
Knowledge distillation approach for ECG based sleep staging assisted by EEG based sleep staging

## Research

### Experimental Architecture

<p align="center">
  <image src = 'images/Architecture.png' >
</p>

Knowledge distillation from EEG to ECG for sleep staging is incorporated here.

### Datasets

Montreal Archive of Sleep Studies (MASS) - Complete 200 subject data used.
- SS1 and SS3 subsets follow AASM guidelines
- SS2, SS4, SS5 subsets follow R_K guidelines

### Directory Structure
- Dataset Spliting
```
├─ Dataset_split
   ├── Data_split_3class_AllData30s_R_K.py
   ├── Data_split_3class_AllData_AASM.py
   ├── Data_split_AllData_30s_R_K.py
   └── Data_split_All_Data_AASM.py
```
- 3 Class Classification
```
├── 3_class
│   ├── datasets
│   │   ├── __init__.py
│   │   └── mass.py
│   │   
│   ├── models
│   │   ├── ecg_base.py
│   │   ├── eeg_base.py
│   │   ├── FEAT_TEMP.py
│   │   ├── FEAT_TRAINING.py
│   │   ├── FEAT_WCE.py
│   │   ├── __init__.py
│   │   └── KD_TEMP.py
│   │   
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── arg_utils.py
│       ├── callback_utils.py
│       ├── dataset_utils.py
│       ├── __init__.py
│       └── model_utils.py
```
- 4 Class Classification
```       
├── 4_class
│   ├── datasets
│   │   ├── __init__.py
│   │   └── mass.py
│   │
│   ├── models
│   │   ├── ecg_base.py
│   │   ├── eeg_base.py
│   │   ├── FEAT_TEMP.py
│   │   ├── FEAT_TRAINING.py
│   │   ├── FEAT_WCE.py
│   │   ├── __init__.py
│   │   └── KD_TEMP.py
│   │   
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── arg_utils.py
│       ├── callback_utils.py
│       ├── dataset_utils.py
│       ├── __init__.py
│       └── model_utils.py
```

