import datasets


def get_data(args):
    # import pdb;pdb.set_trace()
    dm = datasets.available_datasets[args.dataset_type](**vars(args))
    dm.setup("fit")
    dm.setup("test")

    try:
        args.cb_weights = dm.train.dataset.cb_weights
    except AttributeError:
        pass

    return dm, args
