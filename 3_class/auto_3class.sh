echo "-------------------Automating the DL run--------------------"
LOGS_BASE_PATH='/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Staging_KD/3_class/lightning_logs'
TRAIN_PATH='/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Staging_KD/3_class/train.py'


echo "---------------------Baseline Models Training Begins ----------------"
############################# ECG BASE MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "ecg_base" --model_ckpt_name "ECG_BASE_""
python ${TRAIN_PATH} --model_type "ecg_base" --model_ckpt_name "ECG_BASE_" --max_epochs 150
############################# EEG BASE MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "eeg_base" --model_ckpt_name "EEG_BASE_""
python ${TRAIN_PATH} --model_type "eeg_base" --model_ckpt_name "EEG_BASE_" --max_epochs 150
echo "---------------------Baseline Model Training Complete ----------------"


echo "---------------------EEG_Baseline Checkpoint Selection Begin ----------------"
ls -t ${LOGS_BASE_PATH} | grep -P 'version_\d+' | head -n 1 > version_path.txt
EEG_BASE_LOGS_PATH=${LOGS_BASE_PATH}'/'$(cat version_path.txt)'/checkpoints'
echo ${EEG_BASE_LOGS_PATH}
ls ${EEG_BASE_LOGS_PATH} > chkpt_op_1.txt
ls ${EEG_BASE_LOGS_PATH}| grep -Po '\d.\d+' > chkpt_op.txt
i=1
declare -a ck=()
declare -a path=()

while read line; do
        #echo ${line}	
	ck+=(${line})
done < chkpt_op.txt 

while read line; do
        #echo ${line}	
	path+=(${line})
done < chkpt_op_1.txt 

## Higher CK to be choosen from EEG baseline ckpt ##
if python -c "exit(0 if ${ck[0]} > ${ck[1]} else 1)"; then
	   EEG_BASE_CHKP_PATH=${EEG_BASE_LOGS_PATH}'/'${path[0]} 
    else
	   EEG_BASE_CHKP_PATH=${EEG_BASE_LOGS_PATH}'/'${path[1]} 
fi
echo "The checkpoint path is: ${EEG_BASE_CHKP_PATH}"
echo "---------------------EEG_Baseline Checkpoint Selection Completed ----------------"


echo "---------------------KD_TEMP Model Training Begins ----------------"
############################# KD_TEMP MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "kd_temp" --model_ckpt_name "KD_TEMP_""
python ${TRAIN_PATH} --model_type "kd_temp" --model_ckpt_name "KD_TEMP_" --max_epochs 150 --eeg_baseline_path ${EEG_BASE_CHKP_PATH}
echo "---------------------KD_TEMP Model Training Complete ----------------"


echo "--------------------- Feature Training Begins ----------------"
############################# FEATURE TRAINING MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "feat_train" --model_ckpt_name "FEAT_TRAIN_""
python ${TRAIN_PATH} --model_type "feat_train" --model_ckpt_name "FEAT_TRAIN_" --ckpt_monitor "val_Feature_Loss" --ckpt_mode "min" --max_epochs 150 --eeg_baseline_path ${EEG_BASE_CHKP_PATH}
echo "--------------------- Feature Training Complete ----------------"


echo "--------------------- Feature Trained Checkpoint Selection Begin ----------------"
ls -t ${LOGS_BASE_PATH} | grep -P 'version_\d+' | head -n 1 > version_path.txt
FEAT_TRAIN_LOGS_PATH=${LOGS_BASE_PATH}'/'$(cat version_path.txt)'/checkpoints'
echo ${FEAT_TRAIN_LOGS_PATH}
ls ${FEAT_TRAIN_LOGS_PATH} > chkpt_op_1.txt
ls ${FEAT_TRAIN_LOGS_PATH}| grep -Po '\d.\d+' > chkpt_op.txt
i=1
declare -a err=()
declare -a path=()

while read line; do
        #echo ${line}	
	err+=(${line})
done < chkpt_op.txt 

while read line; do
        #echo ${line}	
	path+=(${line})
done < chkpt_op_1.txt 

## Lower loss to be choosen from feature_training ckpt ##
if python -c "exit(0 if ${err[0]} < ${err[1]} else 1)"; then
	   FEAT_TRAIN_CHKP_PATH=${FEAT_TRAIN_LOGS_PATH}'/'${path[0]} 
    else
	   FEAT_TRAIN_CHKP_PATH=${FEAT_TRAIN_LOGS_PATH}'/'${path[1]} 
fi
echo "The checkpoint path is: ${FEAT_TRAIN_CHKP_PATH}"
echo "---------------------Feature Trained Checkpoint Selection Complete ----------------"


echo "--------------------- FEAT_WCE Training Begins ----------------"
############################# FEAT_WCE MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "feat_wce" --model_ckpt_name "FEAT_WCE_""
python ${TRAIN_PATH} --model_type "feat_wce" --model_ckpt_name "FEAT_WCE_" --max_epochs 150 --feat_path ${FEAT_TRAIN_CHKP_PATH}
echo "---------------------FEAT_WCE Training Complete ----------------"


echo "--------------------- FEAT_TEMP Training Begins ----------------"
############################# FEAT_TEMP MODEL ###############################
echo "python ${TRAIN_PATH} --model_type "feat_wce" --model_ckpt_name "FEAT_TEMP_""
python ${TRAIN_PATH} --model_type "feat_temp" --model_ckpt_name "FEAT_TEMP_" --max_epochs 150 --feat_path ${FEAT_TRAIN_CHKP_PATH}
echo "---------------------FEAT_TEMP Training Complete ----------------"