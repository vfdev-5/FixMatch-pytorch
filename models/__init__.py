import torch.nn as nn

from torchvision import models as tv_models

from models.wrn import WideResNet


def setup_ema(ema_model, ref_model):
    ema_model.load_state_dict(ref_model.state_dict())
    for param in ema_model.parameters():
        param.detach_()
    # set EMA model's BN buffers as base model BN buffers:
    for m1, m2 in zip(ref_model.modules(), ema_model.modules()):
        if isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.BatchNorm2d):
            m2.running_mean = m1.running_mean
            m2.running_var = m1.running_var


def setup_model(name, num_classes):
    if name == "WRN-28-2":
        model = WideResNet(num_classes=num_classes)
    else:
        if name in tv_models.__dict__:
            fn = tv_models.__dict__[name]
        else:
            raise RuntimeError("Unknown model name {}".format(name))
        model = fn(num_classes=num_classes)

    return model
