from argparse import ArgumentParser
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from functools import reduce


####################  Building blocks of the network ###########
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
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
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(self.maxpool_kernels
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
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size
            ),
        )

    def forward(self, x):
        shortcuts = []
        features = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)
            features.append(x)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts, features


class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10], in_channels=256, out_channels=5, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
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
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])
        # fmt: off

    def forward(self, z, shortcuts):
        features = []
        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):
            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)
            features.append(z)
        return z, features

class SegmentClassifier(nn.Module):
    def __init__(self, sampling_frequency=200, num_classes=3, epoch_length=30):
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


 
class UTimeModel(nn.Module):
    # def __init__(
    #     self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5,
    #     dilation=2, sampling_frequency=128, num_classes=5, epoch_length=30, lr=1e-4, batch_size=12,
    #     n_workers=0, eval_ratio=0.1, data_dir=None, n_jobs=-1, n_records=-1, scaling=None, **kwargs
    # ):
    def __init__(
        self,
        hparams
    ):
        super().__init__()
        self.hparams = hparams
        # self.save_hyperparameters()
        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
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


    def forward(self, x, resolution=30):
        # Run through encoder
        z, shortcuts, features_enc = self.encoder(x)
        # # Extract Feature after encoder
        features_bottleneck = z
        # Run through decoder
        z, features_dec = self.decoder(z, shortcuts)
        # Extract Feature after decoder
        features_after_dec = z
        # Run dense modeling
        z = self.dense(z)
        features_dense = z
        # # Extract Feature from the final layer
        resolution_samples = self.hparams.sampling_frequency * resolution
        z = z.unfold(-1, resolution_samples, resolution_samples) \
            .mean(dim=-1)
        z = self.segment_classifier(z)

        feature_list = [features_enc,features_bottleneck,features_dec,features_after_dec, features_dense]
        
        return z, feature_list

class FEAT_TRAINING_model(LightningModule):
    def __init__(
        self,
        filters=None,
        in_channels=None,
        maxpool_kernels=None,
        kernel_size=None,
        dilation=None,
        num_classes=None,
        sampling_frequency=None,
        epoch_length=None,
        data_dir=None,
        eeg_baseline_path = None,
        n_jobs=None,
        lr=None,
        train_weights=None,
        val_weights=None,
        test_weights=None,
        *args,
        **kwargs
        ):
        
        super().__init__()
       
        self.train_weights = train_weights
        self.val_weights = val_weights
        self.test_weights = test_weights
        self.epoch_length = epoch_length
        self.save_hyperparameters()

        ## Loading EEG pretrained model and freezed
        self.eeg_baseline = eeg_baseline_path
        PATH_eeg_baseline = self.eeg_baseline
        self.model_eeg = UTimeModel(self.hparams)
        checkpoint_eeg = torch.load(PATH_eeg_baseline)
        checkpoint_eeg['state_dict'].pop('loss.weight')
        self.model_eeg.load_state_dict(checkpoint_eeg['state_dict'])
        self.model_eeg.eval()
        for param in self.model_eeg.parameters():
            param.requires_grad = False
        # Below codeFor checking if ckpt called correctly
        # print("EEG_Model's state_dict:")
        # for param_tensor in self.model_eeg.state_dict():
        #     print(param_tensor, "\t", self.model_eeg.state_dict()[param_tensor].size())
        #########################################################

        ### ECG model not pretrained ####
        self.model_ecg = UTimeModel(self.hparams)
        # Below codeFor checking if ckpt called correctly
        # print("ECG_Model's state_dict:")
        # for param_tensor in self.model_ecg.state_dict():
        #     print(param_tensor, "\t", self.model_ecg.state_dict()[param_tensor].size())
        ####################################################################################

        # Create Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
        )

        # Create Optimizer params
        self.optimizer_params = dict(lr=self.hparams.lr)

        # initializing metrics
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
    
    def forward(self, x_eeg, x_ecg):

        self.model_eeg.eval()  ### PUTS EEG MODEL IN EVAL MODE
        for param in self.model_eeg.parameters():
            param.requires_grad = False
        output_eeg, feature_list_eeg = self.model_eeg(x_eeg)
        output_ecg, feature_list_ecg = self.model_ecg(x_ecg)

        return output_eeg, feature_list_eeg, output_ecg, feature_list_ecg 
    
    
    def classify_segments(self, x_eeg, x_ecg):

        # Run through complete model
        z_eeg, feature_list_eeg, z_ecg, feature_list_ecg = self(x_eeg, x_ecg)

        return z_eeg, feature_list_eeg, z_ecg, feature_list_ecg


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.optimizer_params
        )
    
    def training_step(self, batch_train, batch_idx):
        [eeg_train,ecg_train,y_train] = batch_train
        ## Choose modalities to train
        x_train_eeg = eeg_train.unsqueeze(1)
        x_train_ecg = ecg_train.unsqueeze(1) 
        
        y_train = torch.nn.functional.one_hot(y_train.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        pred_train_eeg, feature_list_eeg, pred_train_ecg, feature_list_ecg  = self.classify_segments(x_train_eeg.float(), x_train_ecg.float())
        
        class_weights = self.train_weights

        train_feature_loss, train_loss_eeg, train_loss_ecg, pred_train_eeg, pred_train_ecg, y_train = self.compute_loss(pred_train_eeg, feature_list_eeg, pred_train_ecg, feature_list_ecg, y_train, class_weights)
        
        self.pred_train_acc = torch.cat([self.pred_train_acc,pred_train_ecg], dim = 0)
        self.y_train_acc = torch.cat([self.y_train_acc,y_train], dim = 0)
        

        ## Logging losses
        self.log('train_loss', train_loss_ecg,                  on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_Feature_Loss', train_feature_loss,      on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_loss_eeg', train_loss_eeg,              on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        return {
            'loss':train_feature_loss,
            'train_Feature_Loss': train_feature_loss,
        }


    def training_epoch_end(self, training_step_outputs):
        
        pred_train_total = self.pred_train_acc
        y_train_total = self.y_train_acc

        print(self.train_conf_matrix_accumulated(pred_train_total.squeeze(1), torch.argmax(y_train_total, dim = 2).squeeze(1)))
        # train_CK_accumulated = self.train_cohenkappa_accumulated(pred_train_total.squeeze(1),torch.argmax(y_train_total, dim = 2).squeeze(1))
        train_F1_accumulated = self.train_f1_accumulated(pred_train_total.squeeze(1),torch.argmax(y_train_total, dim = 2).squeeze(1))
        train_sklearn_accuracy = accuracy_score(torch.argmax(pred_train_total, dim = 2).squeeze(1).cpu().numpy(),torch.argmax(y_train_total, dim = 2).squeeze(1).cpu().numpy())
        
        accuracy = self.train_acc_stages(pred_train_total, y_train_total.int())
        acc_dict = {'W_train_acc':accuracy[0], 'NREM_train_acc':accuracy[1], 'R_train_acc':accuracy[2]}
        f1_score = self.train_f1_stages(pred_train_total.squeeze(1), torch.argmax(y_train_total, dim = 2).squeeze(1))
        f1_dict = {'W_train_f1':f1_score[0], 'NREM_train_f1':f1_score[1], 'R_train_f1':f1_score[2]}

        # self.log('train_CK_accumulated', train_CK_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_F1_accumulated', train_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_acc_sklearn_accumulated', train_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_train_acc = torch.Tensor([]).cuda()
        self.y_train_acc = torch.Tensor([]).cuda()
        
        # pass
    
    def validation_step(self, batch_eval, batch_idx):

        [eeg_val,ecg_val,y_val]= batch_eval
        
        ## Choose modalities to eval
        x_val_eeg = eeg_val.unsqueeze(1)
        x_val_ecg = ecg_val.unsqueeze(1)

        y_val = torch.nn.functional.one_hot(y_val.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        
        pred_val_eeg, feature_list_eeg, pred_val_ecg, feature_list_ecg  = self.classify_segments(x_val_eeg.float(), x_val_ecg.float())
        
        class_weights = self.val_weights
        
        val_feature_loss, val_loss_eeg, val_loss_ecg, pred_val_eeg, pred_val_ecg, y_val = self.compute_loss(pred_val_eeg, feature_list_eeg,  pred_val_ecg, feature_list_ecg, y_val,class_weights)
        
        self.pred_val_acc = torch.cat([self.pred_val_acc,pred_val_ecg], dim = 0)
        self.y_val_acc = torch.cat([self.y_val_acc,y_val], dim = 0)

        ## Logging metrics
        self.log('val_loss', val_loss_ecg,                  on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_Feature_Loss', val_feature_loss,      on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss_eeg', val_loss_eeg,              on_step=True,  on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        return {
            'val_Feature_Loss': val_feature_loss,
        }
    
    
    
    def validation_epoch_end(self, val_step_outputs):
        pred_val_total = self.pred_val_acc
        y_val_total = self.y_val_acc

        print(self.val_conf_matrix_accumulated(pred_val_total.squeeze(1), torch.argmax(y_val_total, dim = 2).squeeze(1)))
        # val_CK_accumulated = self.val_cohenkappa_accumulated(pred_val_total.squeeze(1),torch.argmax(y_val_total, dim = 2).squeeze(1))
        val_F1_accumulated = self.val_f1_accumulated(pred_val_total.squeeze(1),torch.argmax(y_val_total, dim = 2).squeeze(1))
        val_sklearn_accuracy = accuracy_score(torch.argmax(pred_val_total, dim = 2).squeeze(1).cpu().numpy(),torch.argmax(y_val_total, dim = 2).squeeze(1).cpu().numpy())
        
        accuracy = self.val_acc_stages(pred_val_total,y_val_total.int())
        acc_dict = {'W_val_acc':accuracy[0], 'NREM_val_acc':accuracy[1], 'R_val_acc':accuracy[2]}
        f1_score = self.val_f1_stages(pred_val_total.squeeze(1),torch.argmax(y_val_total, dim = 2).squeeze(1))
        f1_dict = {'W_val_f1':f1_score[0], 'NREM_val_f1':f1_score[1], 'R_val_f1':f1_score[2]}

        # self.log('val_CK_accumulated', val_CK_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_F1_accumulated', val_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc_sklearn_accumulated', val_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_val_acc = torch.Tensor([]).cuda()
        self.y_val_acc = torch.Tensor([]).cuda()
        # pass
    
    def test_step(self, batch_test, batch_idx):
        
        [eeg_test,ecg_test, y_test] = batch_test
        
        ## Choose modalities to test
        x_test_eeg = eeg_test.unsqueeze(1)
        x_test_ecg = ecg_test.unsqueeze(1)

        y_test = torch.nn.functional.one_hot(y_test.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        pred_test_eeg, feature_list_eeg, pred_test_ecg, feature_list_ecg  = self.classify_segments(x_test_eeg.float(), x_test_ecg.float())
        
        class_weights = self.test_weights
        
        test_feature_loss, test_loss_eeg, test_loss_ecg, pred_test_eeg, pred_test_ecg, y_test = self.compute_loss(pred_test_eeg, feature_list_eeg,  pred_test_ecg, feature_list_ecg, y_test, class_weights)
        

        self.pred_test_acc = torch.cat([self.pred_test_acc,pred_test_ecg], dim = 0)
        self.y_test_acc = torch.cat([self.y_test_acc,y_test], dim = 0)

        ## Metric ##
        # test_CK = self.test_cohenkappa(pred_test_ecg.squeeze(1), torch.argmax(y_test, dim = 2).squeeze(1))
        # print(test_CK)
        print(self.test_conf_matrix(pred_test_ecg.squeeze(1), torch.argmax(y_test, dim = 2).squeeze(1)))

        ## Logging metrics
        self.log('test_loss', test_loss_ecg,                    on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_Feature_Loss', test_feature_loss,        on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_loss_eeg', test_loss_eeg,                on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # y_1s_eeg, feature_dict_eeg, y_1s_ecg, feature_dict_ecg = self.classify_segments(x_test_eeg.float(), x_test_ecg.float(), resolution=1)
        
        return {
            'test_Feature_Loss': test_feature_loss,
            # 'logits_eeg': y_1s_eeg,
            # 'logits_ecg': y_1s_ecg
        }
    
    def test_epoch_end(self, test_step_outputs):
        pred_test_total = self.pred_test_acc
        y_test_total = self.y_test_acc

        print(self.test_conf_matrix_accumulated(pred_test_total.squeeze(1), torch.argmax(y_test_total, dim = 2).squeeze(1)))
        # test_CK_accumulated = self.test_cohenkappa_accumulated(pred_test_total.squeeze(1),torch.argmax(y_test_total, dim = 2).squeeze(1))
        test_F1_accumulated = self.test_f1_accumulated(pred_test_total.squeeze(1),torch.argmax(y_test_total, dim = 2).squeeze(1))
        test_sklearn_accuracy = accuracy_score(torch.argmax(pred_test_total, dim = 2).squeeze(1).cpu().numpy(),torch.argmax(y_test_total, dim = 2).squeeze(1).cpu().numpy())
        
        accuracy = self.test_acc_stages(pred_test_total, y_test_total.int())
        acc_dict = {'W_test_acc':accuracy[0], 'NREM_test_acc':accuracy[1], 'R_test_acc':accuracy[2]}
        f1_score = self.test_f1_stages(pred_test_total.squeeze(1),torch.argmax(y_test_total, dim = 2).squeeze(1))
        f1_dict = {'W_test_f1':f1_score[0], 'NREM_test_f1':f1_score[1], 'R_test_f1':f1_score[2]}

        # self.log('test_CK_accumulated', test_CK_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_F1_accumulated', test_F1_accumulated,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_acc_sklearn_accumulated', test_sklearn_accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(acc_dict,                             on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(f1_dict,                              on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Resetting Storage Tensors after every epoch
        self.pred_test_acc = torch.Tensor([]).cuda()
        self.y_test_acc = torch.Tensor([]).cuda()
        
        # pass
    
    def compute_loss(self, y_pred_eeg, y_feature_list_eeg, y_pred_ecg, y_feature_list_ecg, y_true, class_weights):
        
        if y_pred_eeg.shape[-1] != self.hparams.num_classes:
            y_pred_eeg = y_pred_eeg.permute(dims=[0, 2, 1])
        if y_pred_ecg.shape[-1] != self.hparams.num_classes:
            y_pred_ecg = y_pred_ecg.permute(dims=[0, 2, 1])
        if y_true.shape[-1] != self.hparams.num_classes:
            y_true = y_true.permute(dims=[0, 2, 1])

        ## WEIGHTED CROSS_ENTROPY LOSS #####
        self.wce_loss = nn.CrossEntropyLoss(weight = class_weights, reduction= 'mean')
        
        # Base Loss #
        loss_eeg = self.wce_loss(y_pred_eeg.squeeze(1), torch.argmax(y_true, dim = 2).squeeze(1))
        loss_ecg = self.wce_loss(y_pred_ecg.squeeze(1), torch.argmax(y_true, dim = 2).squeeze(1))
        
      
        ## Attention Loss for Feature Loss ###
        all_layer_loss = []
        for i in range(len(y_feature_list_eeg)):
            
            if isinstance(y_feature_list_eeg[i], list):
                outputT_feat = [torch.sum(x**2,dim=1) for x in y_feature_list_eeg[i]]
                outputS_feat = [torch.sum(x**2,dim=1) for x in y_feature_list_ecg[i]]

                outputT_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputT_feat]
                outputS_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputS_feat]

                outputT_feat = [x.unsqueeze(1) for x in outputT_feat]
                outputS_feat = [x.unsqueeze(1) for x in outputS_feat]

                feat_l1_loss = [F.l1_loss(x,y) for x,y in zip(outputT_feat,outputS_feat)]

                loss = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss)
            else:
                outputT_feat = torch.sum(y_feature_list_eeg[i]**2,dim=1)  
                outputS_feat = torch.sum(y_feature_list_ecg[i]**2,dim=1)

                outputT_feat = torch.nn.functional.normalize(outputT_feat, dim=0)
                outputS_feat = torch.nn.functional.normalize(outputS_feat,dim=0)

                outputT_feat = outputT_feat.unsqueeze(1)
                outputS_feat = outputS_feat.unsqueeze(1)

                feat_l1_loss = [F.l1_loss(x,y) for x,y in zip(outputT_feat,outputS_feat)]

                loss = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss)

            all_layer_loss.append(loss)
        # feat_no = 1    ### 0=encoder layers, 1=bottleneck, 2=decoder layers, 3=afterdecoder, 4=dense
        # feat_loss = all_layer_loss[feat_no]  ### selective feat
        feat_loss = sum(all_layer_loss)/len(all_layer_loss)   #### ALL features combined
        
        return feat_loss, loss_eeg, loss_ecg, y_pred_eeg, y_pred_ecg, y_true
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
        architecture_group.add_argument('--sampling_frequency', default=200, type=int)
        architecture_group.add_argument('--num_classes', default=3, type=int)
        architecture_group.add_argument('--epoch_length', default=30, type=int)
        architecture_group.add_argument("--eeg_baseline_path", default=None, help= 'Enter checkpoint path to trained features from Feature_Training')
        # /media/Sentinel_2/Pose2/Vaibhav/MASS_CODE/Sleep_Joint_Learning/MODULED/lightning_logs/version_10/checkpoints/EEG_BASE100epoch=01-val_CK_accumulated=0.7393.ckpt
        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group('optimizer')
        optimizer_group.add_argument('--lr', default=1e-3, type=float)

        return parser

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from datasets.mass import MassDataModule
    from pytorch_lightning.core.memory import ModelSummary

    parser = ArgumentParser(add_help=False)
    parser = MassDataModule.add_dataset_specific_args(parser)
    parser = FEAT_TRAINING_model.add_model_specific_args(parser)
    args = parser.parse_args()

    utime =FEAT_TRAINING_model(vars(args))