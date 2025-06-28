import torch.nn as nn

def weights_init_normal(model_layer):
    classname = model_layer.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model_layer.weight.data, mean=0.0, std=0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(model_layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(model_layer.bias.data, val=0.0)