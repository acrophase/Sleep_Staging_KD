import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
import utils
import numpy as np
import random

torch.backends.cudnn.benchmark = False

def args_help():

    ''' Document to help with args'''
    print('General arguments required' )
    print( '--dataset_type(str): "mass"' )
    print( '--model_type(str): Any model from Models folder' )
    print( '--model_ckpt_name(str): Required name for ckpt as per model type' )
    print( '--ckpt_monitor(str): Metric to be monitor for ckpt saving Default=val_F1_accumulated')
    print( '--ckpt_mode(str): "min" or "max" mode for ckpt saving  Default=max')
    
    ### For FEAT_TRAINING and KD_TEMP codes
    print( '--eeg_baseline_model(str): Path to eeg-baseline-ckpt' )
    ### For FEAT_WCE and FEAT_TEMP codes
    print( '--feat_path(str): Path to feat-trained model ckpt' )

def run_training(args):
    
    # Remember to seed!
    seed_everything(0, workers=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Setup data module for training
    dm, args = utils.get_data(args)

    # Setup model
    model = utils.get_model(args)

    # Setup callbacks    
    callbacks = utils.get_callbacks(ckpt_monitor = args.ckpt_monitor, 
                                    ckpt_name = args.model_ckpt_name, 
                                    mode = args.ckpt_mode)

    trainer = Trainer(
                deterministic=True,
                callbacks= callbacks,
                min_epochs=1, 
                max_epochs= args.max_epochs,
                check_val_every_n_epoch=2,
                gpus=1,
                progress_bar_refresh_rate=1,
                num_sanity_val_steps=0,
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
            model.load_state_dict(checkpoint_trained_feat['state_dict'])
    #####################################################

    ### TRAINING from the CODE ###
    trainer.fit(model, train_loader, val_loader)
    print(callbacks.best_model_path)
    trainer.test(dataloaders=test_dataloader,ckpt_path="best",verbose=True)

if __name__ == "__main__":
    args_help()
    args = utils.get_args()
    run_training(args)