"""
YOLO模型架构
"""

import torch.nn as nn
from .utils.parse_config import parse_model_config

def create_modules(module_defs):
    if module_defs[0]["type"] == "net":
        hyperparams = module_defs.pop(0)
    else:
        raise ValueError("Incorrect configuration settings, lack of hyperparameters, or incorrect type names")

    # outputs_channel记录各层网络层的输出通道数
    outputs_channel = [int(hyperparams["channels"])] # 获取输入层的输出通道数
    
    module_list = nn.ModuleList()
    
    # 'type': {'convolutional', 'yolo', 'route', 'net', 'shortcut', 'upsample'}
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        channel = outputs_channel[-1]
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            channel = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad = (kernel_size - 1) // 2 if module_def["pad"] else 0 # 维持卷积前后图片大小不变
            
            modules.add_module(
                name=f"conv_{i}",
                module=nn.Conv2d(
                    in_channels=outputs_channel[-1],
                    out_channels=channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    bias=not bn # 当带有 BN 层时, 会抵消掉前一层的偏置项
                )
            )
            
            if bn:
                modules.add_module(
                    name=f"batch_norm_{i}",
                    module=nn.BatchNorm2d(channel)
                )
            
            if module_def["activation"] == "leaky":
                modules.add_module(
                    name=f"leaky_{i}",
                    module=nn.LeakyReLU(negative_slope=0.1)
                )
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            
            if kernel_size == 2 and stride == 1:
                # (kernel_size - 1) // 2 会得到 0，这意味着没有填充，导致卷积后的特征图尺寸缩小
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module(
                    f"_debug_padding_{i}", 
                    padding
                )
                
            modules.add_module(
                name=f"maxpool_{i}",
                module=nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2
                )
            )
        elif module_def["type"] == "upsample":
            stride = int(module_def["stride"])
            
            modules.add_module(
                name=f"upsample_{i}",
                module=nn.Upsample(
                    scale_factor=stride,
                    mode="nearest"
                )
            )
        elif module_def["type"] == "route":
            layers = [int(layer) for layer in module_def["layers"].split(',')]
            channel = sum(outputs_channel[layer] for layer in layers)
            modules.add_module(
                name=f"route_{i}",
                module=EmptyLayer()
            )
        elif module_def["type"] == "shortcut":
            channel = outputs_channel[int(module_def["from"])]
            modules.add_module(
                name=f"shortcut_{i}",
                module=EmptyLayer()
            )
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(idx.strip()) for idx in module_def["mask"].split(",")]
            anchors = [int(anchor.strip()) for anchor in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[idx] for idx in anchor_idxs]
            
            pred_num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            
            # 定义 Detection Layer
            modules.add_module(
                name=f"yolo_{i}",
                module=YOLOLayer(anchors, pred_num_classes, img_height)
            )
            
        module_list.append(modules)
        outputs_channel.append(channel)
        
    return hyperparams, module_list
        
        
class EmptyLayer(nn.Module):
    """起到占位符(placeholder)的作用"""
    def __init__(self) -> None:
        super().__init__()
        

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim) -> None:
        super().__init__()
