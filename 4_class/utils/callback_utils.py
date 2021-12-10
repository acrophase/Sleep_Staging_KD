# from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping


def get_callbacks(ckpt_monitor, ckpt_name, mode):

    try:
        checkpoint_callback = ModelCheckpoint(
                                        # monitor='val_CK_accumulated',
                                        monitor=ckpt_monitor,
                                        filename= ckpt_name + '{epoch:02d}-{val_F1_accumulated:.4f}',
                                        verbose=True,
                                        auto_insert_metric_name= True,
                                        save_top_k=2,
                                        mode=mode,
                                        save_on_train_epoch_end = False,
                                        save_last=True,
                                        )

        # feat_checkpoint_callback = ModelCheckpoint(
        #                                 monitor='val_Feature_Loss',
        #                                 filename=ckpt_name + '{epoch:02d}-{val_Feature_Loss:.5f}',
        #                                 verbose=True,
        #                                 auto_insert_metric_name= True,
        #                                 save_top_k=2,
        #                                 mode='min',
        #                                 save_on_train_epoch_end = False,
        #                                 )

        # # Setup callback(s) params
        # checkpoint_monitor_params = dict(
        #     filepath=os.path.join(args.save_dir, "{epoch:03d}-{eval_loss:.2f}"),
        #     monitor=args.checkpoint_monitor,
        #     save_last=True,
        #     save_top_k=1,
        # )
        # earlystopping_parameters = dict(monitor=args.earlystopping_monitor, patience=args.earlystopping_patience,)
        # callbacks = [
        #     pl_callbacks.ModelCheckpoint(**checkpoint_monitor_params),
        #     pl_callbacks.EarlyStopping(**earlystopping_parameters),
        #     pl_callbacks.LearningRateMonitor(),
        # ]

        return checkpoint_callback #feat_checkpoint_callback

    except AttributeError:
        return None, None
