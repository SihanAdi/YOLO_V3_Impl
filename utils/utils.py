import torch.nn as nn
import torch

def weights_init_normal(model_layer):
    classname = model_layer.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model_layer.weight, mean=0.0, std=0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(model_layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(model_layer.bias, val=0.0)
        
def bbox_iou(box1, box2, x1y1x2y2=True, epsilon=1e-16):
    """计算bbox的IOU"""
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area + epsilon)
        
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
            
            # box转换；target 的 box 的高宽存储的是相对于图片的宽和高的比例
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gh = target[b, t, 3] * nG
            gw = target[b, t, 4] * nG
            
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            
            # Get shape of gt box
            gt_box = torch.tensor([0, 0, gw, gh], dtype=torch.float32).unsqueeze(0)
            
            # Get shape of anchor box
            anchor_shapes = torch.cat(
                [
                    torch.zeros((len(anchors), 2), dtype=torch.float32),
                    torch.tensor(
                        anchors, dtype=torch.float32
                    )
                ], dim=1
            )
            
            anchor_ious = bbox_iou(gt_box, anchor_shapes)
            # 将交并比大于阈值的部分设置conf_mask的对应位为0(ignore)
            conf_mask[b, anchor_ious > ignore_thres, gj, gi] = 0
            best_anchor_index = torch.argmax(anchor_ious)
            
            gt_box = torch.tensor([gx, gy, gw, gh], dtype=torch.float32).unsqueeze(0)
            # 获取最佳的预测 box
            pred_box = pred_boxes[b, best_anchor_index, gj, gi].unsqueeze(0)
            
            mask[b, best_anchor_index, gj, gi] = 1
            conf_mask[b, best_anchor_index, gj, gi] = 1
            
            # 转换target的坐标
            tx[b, best_anchor_index, gj, gi] = gx - gi
            ty[b, best_anchor_index, gj, gi] = gy - gj
            tw[b, best_anchor_index, gj, gi] = torch.log(gw / anchor_ious[best_anchor_index][0] + 1e-16)
            th[b, best_anchor_index, gj, gi] = torch.log(gh / anchor_ious[best_anchor_index][1] + 1e-16)
            
            target_label = int(target[b, t, 0])
            tcls[b, best_anchor_index, gj, gi, target_label] = 1
            tconf[b, best_anchor_index, gj, gi] = 1
            
            pred_iou = bbox_iou(pred_box, gt_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_anchor_index, gj, gi])
            pred_score = pred_conf[b, best_anchor_index, gj, gi]
            
            if pred_iou > 0.5 and pred_label == target_label and pred_score > 0.5:
                nCorrect += 1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls