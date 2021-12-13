import models


def get_model(args):

    model = models.available_models[args.model_type](**vars(args))

    return model