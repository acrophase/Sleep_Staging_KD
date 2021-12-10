import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
import utils
import argparse

torch.backends.cudnn.benchmark = True

def args_help():

    ''' Document to help with args'''
    print('General arguments required' )
    print( '--dataset_type(str): "mass"' )
    print( '--model_type(str): Any model from Models folder' )
    print( '--model_ckpt_name(str): Required name for ckpt as per model type' )
    print( '--ckpt_monitor(str): Metric to be monitor for ckpt saving   Default=val_F1_accumulated')
    print( '--ckpt_mode(str): "min" or "max" mode for ckpt saving   Default=max')
    
    ### For FEAT_TRAINING and KD_TEMP codes
    print( '--eeg_baseline_model(str): Path to eeg-baseline-ckpt' )
    ### For FEAT_WCE and FEAT_TEMP codes
    print( '--feat_path(str): Path to feat-trained model ckpt' )

'''
## Ckpts from the POCT logs ##
ECG_BASE_ckpt = "/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_69/checkpoints/ECG_base100epoch=29-val_CK_accumulated=0.2607.ckpt"
EEG_BASE_ckpt = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_70/checkpoints/EEG_base100epoch=47-val_CK_accumulated=0.7769.ckpt'
KD_TEMP_ckpt = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_72/checkpoints/ECG_KD_CE_TEMPLOSS(0.2,4)_AASM_epoch=69-val_CK_accumulated=0.247444.ckpt'
FEAT_TRAIN_ckpt = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_78/checkpoints/Feature_Training_ALLDATA30s_epoch=99-val_Feature_Loss=0.02845.ckpt'
FEAT_WCE_ckpt = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_79/checkpoints/ECG_feat_KD_training_ALLDATA_epoch=99-val_CK_accumulated=0.225631.ckpt'
FEAT_TEMP_ckpt = '/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_80/checkpoints/ECG_feat_KD_training_ALLDATA_epoch=93-val_CK_accumulated=0.247235.ckpt'
'''

def run_testing(args):
    
    # Remember to seed!
    seed_everything(42, workers=True)

    # Setup data module for testing
    dm, args = utils.get_data(args)

    # Setup model
    model = utils.get_model(args)

    # Setup callbacks    
    callbacks = utils.get_callbacks(ckpt_monitor = args.ckpt_monitor, 
                                    ckpt_name = args.model_ckpt_name, 
                                    mode = args.ckpt_mode)

    trainer = Trainer(
                # deterministic=True,
                callbacks= callbacks,
                min_epochs=1, 
                max_epochs= args.max_epochs,
                check_val_every_n_epoch=2,
                gpus=1,
                progress_bar_refresh_rate=1,
                )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    #### Assigning CKPT_MONITOR AND MODE FOR FEATURE TRAINING #####
    if args.model_type == "feat_train":
        args.ckpt_monitor = "val_Feature_Loss"
        args.ckpt_mode = "min"
    #####################################################

    ### LOADING FROM Feature CKPT ###
    ''' Use only when Feature trained ckpt to be called 
        i.e in FEAT_WCE and FEAT_TEMP code'''
    for arg in vars(args):
        if arg == 'feat_path':
            PATH_trained_feat = args.feat_path
            checkpoint_trained_feat = torch.load(PATH_trained_feat)
            checkpoint_trained_feat['state_dict'].pop('wce_loss.weight')
            model.load_state_dict(checkpoint_trained_feat['state_dict'])
    #####################################################

    ### TRAINING-TESTING FROM CKPT ###

    PATH_ckpt = args.test_ckpt
    checkpoint_load = torch.load(PATH_ckpt)
    try:
        checkpoint_load['state_dict'].pop('loss.weight')
    except:
        checkpoint_load['state_dict'].pop('wce_loss.weight')

    model.load_state_dict(checkpoint_load['state_dict'])

    ## When TRAINING FROM CKPT
    # trainer.fit(model, train_loader, val_loader)
    # print(checkpoint_callback.best_model_path)
    # trainer.test(dataloaders=test_dataloader,ckpt_path="/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/POCT/lightning_logs/version_35/checkpoints/EEG_base200epoch=69-val_F1_score=0.8144.ckpt",verbose=True)
    
    ### WHEN ONLY TESTING
    print(callbacks.best_model_path)
    trainer.test(model = model, dataloaders=test_dataloader, verbose=True)

if __name__ == "__main__":
    args_help()
    args = utils.get_args()
    run_testing(args)