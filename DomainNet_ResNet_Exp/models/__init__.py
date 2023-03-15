from models.clip_resnet import ModifiedResNet
from models.resnet import resnet50


def get_model(arch, **kwargs):

    if arch == "resnet50":
        return resnet50(**kwargs)
    elif arch == "clip_resnet50":
        return ModifiedResNet(**kwargs)
    else:
        raise ValueError("Model {} not available".format(arch))
