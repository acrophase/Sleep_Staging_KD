import datasets


def get_data(args):
    dm = datasets.available_datasets[args.dataset_type](**vars(args))
    # dm.setup("fit")
    # dm.setup("test")

    return dm, args
