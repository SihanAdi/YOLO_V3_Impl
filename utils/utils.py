import torch.nn as nn
import torch

def weights_init_normal(model_layer):
    classname = model_layer.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model_layer.weight, mean=0.0, std=0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(model_layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(model_layer.bias, val=0.0)
        
def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
    """
    在 YOLO 中，网络并不是直接回归 (bx, by, bw, bh)（即边界框在输入图像上的真实坐标）
        对中心点 (bx, by) 做相对网格位置的偏移预测（归一化到 [0,1]）
        对宽高 (bw, bh) 进行相对于 anchor 的 log-space 比例预测
    输入参数：
    pred_boxes: 网络预测出来的框（尚未解码到像素坐标），格式是 (tx, ty, tw, th)
                尺寸：[Batch Size, Anchor Num, Grid Size, Grid Size, 4]
    pred_conf: 预测框的置信度分数（是否包含物体的概率）
                尺寸：[B, A, G, G]
    pred_cls: 预测框的类别分布
                尺寸：[B, A, G, G, Class num]
    target: Ground Truth 真实标签(class, x, y, w, h)
                尺寸：[B, T, 5]; T: 每张图最多多少个目标（比如 50 个），多余填 0
    anchors: 每个 anchor 的原始宽高
                尺寸：[A, 2]
    """
    nB = target.shape[0]
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    # 每个anchor坐标
    mask = torch.zeros(nB, nA, nG, nG)
    # 每个anchor的置信度
    conf_mask = torch.ones(nB, nA, nG, nG)
    # anchor信息
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)
    
    # 当前这批数据里，真实需要检测出来的所有目标数量
    nGT = 0 
    nCorrect = 0
    for b in range(nB):
        for t in target.shape[1]:
            if target[b,t].sum() == 0:
                # 如果 box 的5个值(从标签到坐标)都为0，则认定该标签无需判定
                continue
            nGT += 1