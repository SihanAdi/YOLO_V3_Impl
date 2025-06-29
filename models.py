"""
YOLO模型架构
"""

import torch.nn as nn
import torch
from .utils.parse_config import parse_model_config
from .utils.utils import build_targets

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
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.num_bbox_attrs = 5 + num_classes
        self.ignore_thresh = 0.5
        self.lambda_coord = 1
        
        # 定义损失
        self.mse_loss = nn.MSELoss(size_average=True) # 计算边界框坐标(x,y,w,h)的损失, 对损失取平均
        self.bce_loss = nn.BCELoss(size_average=True) # 计算目标置信度(confidence score)的损失, 对损失取平均
        self.ce_loss = nn.CrossEntropyLoss() # 计算类别分类的损失
        
    def forward(self, x, targets=None):
        # x: (batch, anchors * (5 + num_classes), w, h)
        nB = x.shape[0]
        nG = x.shape[2]
        nA = self.num_anchors
        stride = self.img_dim / nG
        device = x.device
        prediction = x.view(nB, nA, self.num_bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        
        # 由于预测相对偏移量
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        # 确定每个网格的基准坐标
        grid_x = torch.arange(nG, dtype=torch.float32, device=device).repeat(nG, 1).view([1, 1, nG, nG])
        grid_y = torch.arange(nG, dtype=torch.float32, device=device).repeat(nG, 1).view([1, 1, nG, nG])
        
        # 将原图上 anchors 的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box
        scaled_anchors = torch.tensor(data=[
            (a_w / stride, a_h / stride) for a_w, a_h in self.anchors
        ], dtype=torch.float32, device=device)
        
        anchor_w = scaled_anchors[:, 0:1].view([1, nA, 1, 1])
        anchor_h = scaled_anchors[:, 1:2].view([1, nA, 1, 1])
        
        pred_boxes = torch.zeros(prediction[..., :4].shape, dtype=torch.float32, device=device)
        pred_boxes[..., 0] = x.detach() + grid_x
        pred_boxes[..., 1] = y.detach() + grid_y
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_h
        
        if targets:
            # 训练阶段
            self.mse_loss = self.mse_loss.to(device)
            self.bce_loss = self.bce_loss.to(device)
            self.ce_loss = self.ce_loss.to(device)
            
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.to('cpu').detach(),
                pred_conf=pred_conf.to('cpu').detach(),
                pred_cls=pred_cls.to('cpu').detach(),
                target=targets.to('cpu').detach(),
                anchors=scaled_anchors.to('cpu').detach(),
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thresh,
                img_dim=self.img_dim
            )
            
            num_proposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / num_proposals)
            
            mask = mask.to(dtype=torch.bool, device=device)
            conf_mask = conf_mask.to(dtype=torch.bool, device=device)
            tx = tx.to(dtype=torch.float32, device=device)
            ty = ty.to(dtype=torch.float32, device=device)
            tw = tw.to(dtype=torch.float32, device=device)
            th = th.to(dtype=torch.float32, device=device)
            tconf = tconf.to(dtype=torch.float32, device=device)
            tcls = tcls.to(dtype=torch.long, device=device)
            
            conf_mask_true = mask # mask 只有best_n对应位为1, 其余都为0, 具有最佳匹配度的anchor box
            conf_mask_false = conf_mask - mask # conf_mask中iou大于ignore_thres的为0, 其余为1, best_n也为1, iou小于0.5的anchor box
            
            """计算loss"""
            # 忽略 non-existing objects, 计算相应的位置 loss
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            
            # 计算置信度 loss
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            ) + self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            )
            
            # 计算分类 loss
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[conf_mask_true], torch.argmax(tcls[conf_mask_true], 1))
            
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            # 非训练阶段则直接返回准确率, output的shape为: [nB, -1, 85]
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes)
                 
                ),
                -1
            )
            return output
            