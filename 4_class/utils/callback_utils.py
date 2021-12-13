from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping


def get_callbacks(ckpt_monitor, ckpt_name, mode):

    try:
        checkpoint_callback = ModelCheckpoint(
                                        monitor=ckpt_monitor,
                                        filename= ckpt_name + '{epoch:02d}-{val_F1_accumulated:.4f}',
                                        verbose=True,
                                        auto_insert_metric_name= True,
                                        save_top_k=2,
                                        mode=mode,
                                        save_on_train_epoch_end = False,
                                        save_last=True,
                                        )


        return checkpoint_callback

    except AttributeError:
        return None
