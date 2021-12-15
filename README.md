
# Sleep_Staging_Knowledge Distillation

This codebase implements knowledge distillation approach for ECG based sleep staging assisted by EEG based sleep staging model. Knowledge distillation is incorporated here by softmax distillation and another approach by Attention transfer based feature training. The combination of both is the proposed model.

The code implementation was done with Pytorch-lightning framework inside a docker container. Dependencies to be installed inside the docker can be found in [requirements.txt](https://github.com/HTIC-HPOC/Sleep_Staging_KD/blob/main/requirements.txt)

## RESEARCH
### DATASET

[Montreal Archive of Sleep Studies (MASS)](http://ceams-carsm.ca/en/MASS/) - Complete 200 subject data used.
- SS1 and SS3 subsets follow AASM guidelines
- SS2, SS4, SS5 subsets follow R_K guidelines

### KNOWLEDGE DISTILLATION FRAMEWORK
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

## Run Training
Run train.py from 3-class or 4-class directories

To train baseline models

```bash
  python train.py --model_type <"base model type"> --model_ckpt_name <"ckpt name">
```

To run Knowledge Distillation
- Feature Training
```bash
  python train.py --model_type "feat_train" --model_ckpt_name <"ckpt name"> --eeg_baseline_path <"eeg base ckpt path">
```
- Feat_Temp (AT+SD+CL)
```bash
  python train.py --model_type "Feat_Temp" --model_ckpt_name <"ckpt name"> --feat_path <"path to feature trained ckpt">
```
- Feat_WCE (AT+CL)
```bash
  python train.py --model_type "feat_wce" --model_ckpt_name <"ckpt name"> --feat_path <"path to feature trained ckpt">
```
- KD-Temp (SD+CL)
```bash
  python train.py --model_type "kd_temp" --model_ckpt_name <"ckpt name"> --eeg_baseline_path <"eeg base ckpt path">
```

## Run Testing
Run test.py from 3-class or 4-class directories

To test from checkpoints
```bash
  python test.py --model_type <"model type"> --test_ckpt <"Path to checkpoint>
```
Other arguments can be used for training and testing as per requirements

<h2>
<details> <summary> Reproducing experiments </summary><br/>
<h6>

<!-- Coming Soon ... -->

<ins>
To run the experiment on RTX-3090, the dependencies were saved as a docker image. Follow the below steps to run on RTX-3090
</ins>


 - Pull the docker image from [DockerHub](https://hub.docker.com/repository/docker/vaibhavjoshiiitm/poct2022/general)

```bash
  docker pull vaibhavjoshiiitm/poct2022
```

- Run the docker container
```bash
  sudo docker run --gpus all 
```
  <!-- # --ipc=host -it -v /media/acrophase:/media <"docker_name"> -->

- Run training/testing shown in Run Training / Run Testing section.

<ins>
To run the experiments on OTHER than RTX-3090 GPU Follow the below steps
</ins>
- Navigate to the desired directory and Install requirements

```bash
  pip install -r requirements.txt
```
- Run training/testing shown in Run Training / Run Testing section.

Checkpoints to reproduce the test results can be found in [this link](https://drive.google.com/drive/folders/1Vy_ieBrNydkJ-s20Xg79gpWjVsJzxk8y?usp=sharing)

</details>

## Directory Map

### Dataset Spliting: 
Splits Data in train-val-test for 4-class and 3-class cases (AASM and R_K both)
```
├─ Dataset_split
   ├── Data_split_3class_AllData30s_R_K.py
   ├── Data_split_3class_AllData_AASM.py
   ├── Data_split_AllData_30s_R_K.py
   └── Data_split_All_Data_AASM.py
```
### 3 Class Classification: 
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
### 4 Class Classification: 
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
## Acknowledgements

 - [U-Time model pytorch implementation](https://github.com/neergaard/utime-pytorch)
 - [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
 
## Authors

- [Vaibhav Joshi](https://github.com/VAIBHAV2900)