import argparse
import pytorch_lightning as pl
import datasets
import models


# def get_args(data = None, model_type = None):
def get_args():
    # import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser(add_help=False)
    # Check the supplied model type
    # parser.add_argument("--dataset_type", type=str, default=data)
    parser.add_argument("--dataset_type", default="mass", type=str,help= 'Enter the name of the data')
    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # Check the supplied model type
    parser.add_argument("--model_type", default=None, type=str, help= 'Enter the name of the model')

    ### args realted to callbacks for Trainer
    parser.add_argument("--model_ckpt_name", default=None, type=str, help= 'Enter the name for ckpt saving')
    parser.add_argument("--ckpt_monitor", default= 'val_CK_accumulated', type = str, help= 'Metric to monitor to save the ckpt')
    parser.add_argument("--ckpt_mode", default= 'max', type = str, help= 'Mode for the ckpt_monitor metric to save the ckpt')
    temp_args, _ = parser.parse_known_args()

    # add args from dataset
    parser = datasets.available_datasets[temp_args.dataset_type].add_dataset_specific_args(parser)

    # add args from model
    parser = models.available_models[temp_args.model_type].add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    return args
