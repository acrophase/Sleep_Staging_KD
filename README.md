# Sleep_Staging_Knowledge Distillation
This codebase implements knowledge distillation approach for ECG based sleep staging assisted by EEG based sleep staging model. Knowledge distillation is incorporated here by softmax distillation and another approach by Attention transfer based feature training. The combination of both is the proposed model.

Single-channel ECG based sleep staging is improved as conclusion.

# Research
## Datasets

[Montreal Archive of Sleep Studies (MASS)](http://ceams-carsm.ca/en/MASS/) - Complete 200 subject data used.
- SS1 and SS3 subsets follow AASM guidelines
- SS2, SS4, SS5 subsets follow R_K guidelines

## Knowledge Distillation Framework
 Knowledge distillation framework using minor modifications in [U-Time](https://arxiv.org/abs/1910.11162) as base model.

 
<p align="center">
  <image src = 'images/Architecture.png' >
</p>


Improvement in bottleneck features from ECG_Base model to KD_model as a result of Knowledge distillation compared to EEG_base model features.

<p align="center">
  <image src = 'images/Feature_Plots.png' >
</p>

Case 1 : KD_model predicting correctly, ECG_Base predicting incorrectly

Case 2 : KD_model predicting incorrectly, ECG_Base predicting correctly

# Directory Structure

## Neccessary arguments to run with scripts for training

> --dataset_type(str): "mass"

>--model_type(str): Any model from Models folder

>--model_ckpt_name(str): Required name for ckpt as per model type

>--ckpt_monitor(str): Metric to be monitor for ckpt saving Default=val_F1_accumulated')

>--ckpt_mode(str): "min" or "max" mode for ckpt saving  Default=max')

- For FEAT_TRAINING and KD_TEMP codes
>--eeg_baseline_model(str): Path to eeg-baseline-ckpt' )

- For FEAT_WCE and FEAT_TEMP codes
>--feat_path(str): Path to feat-trained model ckpt' )

## Dataset Spliting: 
Splits Data in train-val-test for 4-class and 3-class cases (AASM and R_K both)
```
├─ Dataset_split
   ├── Data_split_3class_AllData30s_R_K.py
   ├── Data_split_3class_AllData_AASM.py
   ├── Data_split_AllData_30s_R_K.py
   └── Data_split_All_Data_AASM.py
```
## 3 Class Classification: 
Run train.py with neccessary arguments for training 3-class sleep staging
```
├── 3_class
│   ├── datasets
│   │   ├── __init__.py
│   │   └── mass.py
│   │   
│   ├── models
│   │   ├── __init__.py
│   │   ├── ecg_base.py
│   │   ├── eeg_base.py
│   │   ├── FEAT_TEMP.py
│   │   ├── FEAT_TRAINING.py
│   │   ├── FEAT_WCE.py
│   │   └── KD_TEMP.py
│   │   
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── arg_utils.py
│       ├── callback_utils.py
│       ├── dataset_utils.py
│       └── model_utils.py
```
## 4 Class Classification: 
Run train.py with neccessary arguments for training 4-class sleep staging
```       
├── 4_class
│   ├── datasets
│   │   ├── __init__.py
│   │   └── mass.py
│   │
│   ├── models
│   │   ├── __init__.py
│   │   ├── ecg_base.py
│   │   ├── eeg_base.py
│   │   ├── FEAT_TEMP.py
│   │   ├── FEAT_TRAINING.py
│   │   ├── FEAT_WCE.py
│   │   └── KD_TEMP.py
│   │   
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── arg_utils.py
│       ├── callback_utils.py
│       ├── dataset_utils.py
│       └── model_utils.py
```

