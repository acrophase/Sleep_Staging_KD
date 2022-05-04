from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
import random
from pytorch_lightning import LightningModule
import torchmetrics
from sklearn.metrics import accuracy_score


####################  Building blocks of the network ###########
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu", dropout = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.dropout = dropout
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2, dropout = 0):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
                dropout = self.dropout,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
                dropout = self.dropout,
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout = self.dropout,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout = self.dropout,
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10], in_channels=256, out_channels=5, kernel_size=5, dropout = 0):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dropout = dropout
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_kernels[k]),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                activation='relu',
                dropout = self.dropout,
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dropout = self.dropout,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dropout = self.dropout,
            ),
        ) for k in range(self.depth)])
        # fmt: off

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):

            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)

        return z

class SegmentClassifier(nn.Module):
    def __init__(self, sampling_frequency=200, num_classes=4, epoch_length=30):
        super().__init__()
        self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.layers = nn.Sequential(
            # nn.AvgPool1d(kernel_size=(self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1),
            # nn.Softmax(dim=1),
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x):
        return self.layers(x)

##################################################################################################

############ PYTORCH LIGHTNING MODULE FOR ARCHITECTURE ##########
class EEG_BASE_Model(LightningModule):
    # def __init__(
    #     self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5,
    #     dilation=2, sampling_frequency=128, num_classes=5, epoch_length=30, lr=1e-4, batch_size=12,
    #     n_workers=0, eval_ratio=0.1, data_dir=None, n_jobs=-1, n_records=-1, scaling=None, **kwargs
    # ):
    def __init__(
        self,
        filters=None,
        in_channels=None,
        maxpool_kernels=None,
        kernel_size=None,
        dilation=None,
        dropout=None,
        num_classes=None,
        sampling_frequency=None,
        epoch_length=None,
        data_dir=None,
        n_jobs=None,
        lr=None,
        train_weights=None,
        val_weights=None,
        test_weights=None,
        weights_file=None,
        *args,
        **kwargs
    ):

        super().__init__()
        weights_list = []
        with open(weights_file) as f:
            weights_list = f.readlines()
            weights_list = [float(i) for i in weights_list]
        self.train_weights = torch.tensor(weights_list).cuda()
        self.val_weights = torch.tensor(weights_list).cuda()
        self.test_weights = torch.tensor(weights_list).cuda()
        self.epoch_length = epoch_length
        self.save_hyperparameters()
        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
            dropout = self.hparams.dropout,
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
            dropout = self.hparams.dropout,
        )
        self.dense = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.filters[0], out_channels=self.hparams.num_classes, kernel_size=1, bias=True),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)
        self.segment_classifier = SegmentClassifier(
            sampling_frequency=self.hparams.sampling_frequency,
            num_classes=self.hparams.num_classes,
            epoch_length=self.hparams.epoch_length
        )

        # Create Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
        )

        # Create Optimizer params
        self.optimizer_params = dict(lr=self.hparams.lr)

        self.train_acc_stages = torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average=None)
        self.val_acc_stages =  torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average=None)
        self.test_acc_stages =  torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average=None)

        self.test_cohenkappa = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        self.train_cohenkappa_accumulated = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        self.val_cohenkappa_accumulated = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        self.test_cohenkappa_accumulated = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        
        self.train_f1_stages = torchmetrics.F1(num_classes=self.hparams.num_classes, average=None)
        self.val_f1_stages =  torchmetrics.F1(num_classes=self.hparams.num_classes, average=None)
        self.test_f1_stages =  torchmetrics.F1(num_classes=self.hparams.num_classes, average=None)
        self.train_f1_accumulated =  torchmetrics.F1(num_classes=self.hparams.num_classes, average='weighted')
        self.val_f1_accumulated =  torchmetrics.F1(num_classes=self.hparams.num_classes, average='weighted')
        self.test_f1_accumulated =  torchmetrics.F1(num_classes=self.hparams.num_classes, average='weighted')

        self.test_conf_matrix =  torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes)
        self.train_conf_matrix_accumulated = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes)
        self.val_conf_matrix_accumulated = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes)
        self.test_conf_matrix_accumulated = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes)


        self.pred_train_acc = torch.Tensor([]).cuda()
        self.y_train_acc = torch.Tensor([]).cuda()
        self.pred_val_acc = torch.Tensor([]).cuda()
        self.y_val_acc = torch.Tensor([]).cuda()
        self.pred_test_acc = torch.Tensor([]).cuda()
        self.y_test_acc = torch.Tensor([]).cuda()

    def forward(self, x):
        # Run through encoder
        z, shortcuts = self.encoder(x)
        # Run through decoder
        z = self.decoder(z, shortcuts)

        # Run dense modeling
        z = self.dense(z)

        return z

    def classify_segments(self, x, resolution=30):

        # Run through encoder + decoder
        z = self(x)
        # Classify decoded samples
        resolution_samples = self.hparams.sampling_frequency * resolution
        z = z.unfold(-1, resolution_samples, resolution_samples) \
             .mean(dim=-1)
        y = self.segment_classifier(z)

        return y
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.optimizer_params
        )
    
    def training_step(self, batch_train, batch_idx):    
        [eeg_train,ecg_train,y_train] = batch_train
        ## Choose modalities to train
        # x_train = eeg_train.unsqueeze(1)

        y_train = torch.nn.functional.one_hot(y_train.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        eeg_train = eeg_train.squeeze(0).unsqueeze(1)
        y_train = y_train.squeeze(0).squeeze(0)
        pred_train = self.classify_segments(eeg_train.float(), resolution=self.hparams.epoch_length)

        class_weights = self.train_weights

        loss_train,pred_train,y_train = self.compute_loss(pred_train.squeeze(-1), y_train, class_weights)
        
        self.pred_train_acc = torch.cat([self.pred_train_acc,pred_train], dim = 0)
        self.y_train_acc = torch.cat([self.y_train_acc,y_train], dim = 0)

        ## Logging loss
        self.log('train_loss', loss_train,     on_step=True,  on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            'loss': loss_train,
            'pred_train': pred_train,
            'y_train': y_train,
        }
    

    def training_epoch_end(self, training_step_outputs):
        
        pred_train_total = self.pred_train_acc
        y_train_total = self.y_train_acc

        print(self.train_conf_matrix_accumulated(pred_train_total.squeeze(-1), torch.argmax(y_train_total, dim = 1)))
        train_CK_accumulated = self.train_cohenkappa_accumulated(pred_train_total.squeeze(-1),torch.argmax(y_train_total, dim = 1))
        train_F1_accumulated = self.train_f1_accumulated(pred_train_total.squeeze(-1),torch.argmax(y_train_total, dim = 1))
        train_sklearn_accuracy = accuracy_score(torch.argmax(pred_train_total, dim = 1).cpu().numpy(),torch.argmax(y_train_total, dim = 1).cpu().numpy())
        
        accuracy = self.train_acc_stages(pred_train_total, y_train_total.int())
        acc_dict = {'W_train_acc':accuracy[0], 'L_train_acc':accuracy[1], 'D_train_acc':accuracy[2], 'R_train_acc':accuracy[3]}
        f1_score = self.train_f1_stages(pred_train_total, torch.argmax(y_train_total, dim = 1))
        f1_dict = {'W_train_f1':f1_score[0], 'L_train_f1':f1_score[1], 'D_train_f1':f1_score[2], 'R_train_f1':f1_score[3]}

        self.log('train_CK_accumulated', train_CK_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_F1_accumulated', train_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_acc_sklearn_accumulated', train_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_train_acc = torch.Tensor([]).cuda()
        self.y_train_acc = torch.Tensor([]).cuda()
    
    def validation_step(self, batch_eval, batch_idx):
        
        [eeg_val,ecg_val,y_val]= batch_eval
        
        ## Choose modalities to eval
        # x_val = eeg_val.unsqueeze(1)

        y_val = torch.nn.functional.one_hot(y_val.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        eeg_val = eeg_val.squeeze(0).unsqueeze(1)
        y_val = y_val.squeeze(0).squeeze(0)
        pred_val = self.classify_segments(eeg_val.float(), resolution=self.hparams.epoch_length)
        
        class_weights = self.val_weights

        loss_val,pred_val,y_val = self.compute_loss(pred_val.squeeze(-1), y_val, class_weights)

        self.pred_val_acc = torch.cat([self.pred_val_acc,pred_val], dim = 0)
        self.y_val_acc = torch.cat([self.y_val_acc,y_val], dim = 0)

        ## Logging loss
        self.log('val_loss', loss_val,       on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        return {
            'val_loss': loss_val,
            'pred_val': pred_val,
            'y_val': y_val
        }
    


    def validation_epoch_end(self, val_step_outputs):
        pred_val_total = self.pred_val_acc
        y_val_total = self.y_val_acc

        print(self.val_conf_matrix_accumulated(pred_val_total.squeeze(-1), torch.argmax(y_val_total, dim = 1)))
        val_CK_accumulated = self.val_cohenkappa_accumulated(pred_val_total.squeeze(-1),torch.argmax(y_val_total, dim = 1))
        val_F1_accumulated = self.val_f1_accumulated(pred_val_total.squeeze(-1),torch.argmax(y_val_total, dim = 1))
        val_sklearn_accuracy = accuracy_score(torch.argmax(pred_val_total, dim = 1).cpu().numpy(),torch.argmax(y_val_total, dim = 1).cpu().numpy())
        
        accuracy = self.val_acc_stages(pred_val_total,y_val_total.int())
        acc_dict = {'W_val_acc':accuracy[0], 'L_val_acc':accuracy[1], 'D_val_acc':accuracy[2], 'R_val_acc':accuracy[3]}
        f1_score = self.val_f1_stages(pred_val_total,torch.argmax(y_val_total, dim = 1))
        f1_dict = {'W_val_f1':f1_score[0], 'L_val_f1':f1_score[1], 'D_val_f1':f1_score[2], 'R_val_f1':f1_score[3]}

        self.log('val_CK_accumulated', val_CK_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_F1_accumulated', val_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc_sklearn_accumulated', val_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_val_acc = torch.Tensor([]).cuda()
        self.y_val_acc = torch.Tensor([]).cuda()

    def test_step(self, batch_test, batch_idx):
        
        [eeg_test,ecg_test, y_test] = batch_test

        ## Choose modalities to eval
        # x_test=eeg_test.unsqueeze(1)
        
        y_test = torch.nn.functional.one_hot(y_test.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        eeg_test = eeg_test.squeeze(0).unsqueeze(1)
        y_test = y_test.squeeze(0).squeeze(0)
        pred_test = self.classify_segments(eeg_test.float(), resolution=self.hparams.epoch_length)
        
        class_weights = self.test_weights
        
        loss_test,pred_test, y_test = self.compute_loss(pred_test.squeeze(-1), y_test, class_weights)
        

        self.pred_test_acc = torch.cat([self.pred_test_acc,pred_test], dim = 0)
        self.y_test_acc = torch.cat([self.y_test_acc,y_test], dim = 0)

        print(self.test_conf_matrix(pred_test.squeeze(-1), torch.argmax(y_test, dim = 1)))

        ## Logging loss
        self.log('test_loss', loss_test,       on_step=True,  on_epoch=True, prog_bar=True,  logger=True, sync_dist=True)
        
        y_1s = self.classify_segments(eeg_test.float(), resolution=1)
                
        return {
            'test_loss': loss_test,
            'pred_test': pred_test,
            'y_test': y_test,
            'logits': y_1s
        }
    
    def test_epoch_end(self, test_step_outputs):

        pred_test_total = self.pred_test_acc
        y_test_total = self.y_test_acc

        print(self.test_conf_matrix_accumulated(pred_test_total.squeeze(-1), torch.argmax(y_test_total, dim = 1)))
        test_CK_accumulated = self.test_cohenkappa_accumulated(pred_test_total.squeeze(-1),torch.argmax(y_test_total, dim = 1))
        test_F1_accumulated = self.test_f1_accumulated(pred_test_total.squeeze(-1),torch.argmax(y_test_total, dim = 1))
        test_sklearn_accuracy = accuracy_score(torch.argmax(pred_test_total, dim = 1).cpu().numpy(),torch.argmax(y_test_total, dim = 1).cpu().numpy())
        
        accuracy = self.test_acc_stages(pred_test_total, y_test_total.int())
        acc_dict = {'W_test_acc':accuracy[0], 'L_test_acc':accuracy[1], 'D_test_acc':accuracy[2], 'R_test_acc':accuracy[3]}
        f1_score = self.test_f1_stages(pred_test_total,torch.argmax(y_test_total, dim = 1))
        f1_dict = {'W_test_f1':f1_score[0], 'L_test_f1':f1_score[1], 'D_test_f1':f1_score[2], 'R_test_f1':f1_score[3]}

        self.log('test_F1_accumulated', test_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_acc_sklearn_accumulated', test_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_test_acc = torch.Tensor([]).cuda()
        self.y_test_acc = torch.Tensor([]).cuda()
    
    def compute_loss(self, y_pred, y_true, class_weights):

        if y_pred.shape[-1] != self.hparams.num_classes:
            y_pred = y_pred.permute(dims=[0, 2, 1])
        if y_true.shape[-1] != self.hparams.num_classes:
            y_true = y_true.permute(dims=[0, 2, 1])

        self.loss = nn.CrossEntropyLoss(weight = class_weights, reduction= 'mean')
        CE_loss = self.loss(y_pred, torch.argmax(y_true, dim = 1))
        
        return CE_loss,y_pred, y_true

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group('architecture')
        architecture_group.add_argument('--filters', default=[16, 32, 64, 128, 256], nargs='+', type=int)
        architecture_group.add_argument('--in_channels', default=1, type=int)
        architecture_group.add_argument('--maxpool_kernels', default=[3, 2, 2, 2, 2], nargs='+', type=int)
        architecture_group.add_argument('--kernel_size', default=5, type=int)
        architecture_group.add_argument('--dilation', default=2, type=int)
        architecture_group.add_argument('--dropout', default=0.1, type=float)
        architecture_group.add_argument('--sampling_frequency', default=200, type=int)
        architecture_group.add_argument('--num_classes', default=4, type=int)
        architecture_group.add_argument('--epoch_length', default=30, type=int)
        architecture_group.add_argument('--weights_file', default='/media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/MeMeA_Slp_staging_repo/Sleep_Staging_KD/4_class/datasets/class_weights.txt', type=str)
        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group('optimizer')
        optimizer_group.add_argument('--lr', default=1e-3, type=float)

        return parser

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from datasets.mass import MassDataModule
    from datasets.mass_batch import MASSBatchDataModule
    from datasets.mass_batch_small import MASSBatchSmallDataModule
    from pytorch_lightning.core.memory import ModelSummary

    parser = ArgumentParser(add_help=False)
    # parser = MassDataModule.add_dataset_specific_args(parser)
    # parser = MASSBatchDataModule.add_dataset_specific_args(parser)
    parser = MASSBatchSmallDataModule.add_dataset_specific_args(parser)
    parser = EEG_BASE_Model.add_model_specific_args(parser)
    args = parser.parse_args()

    utime = EEG_BASE_Model(vars(args))