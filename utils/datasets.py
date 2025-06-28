"""
读取COCO数据集
"""

from torch.utils.data import Dataset
from PIL import Image, ImageFile
from skimage.transform import resize
import torch
import os
import glob
import numpy as np

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CocoDatasets(Dataset):
    def __init__(self, list_path, img_size=416):
        # 获得图片文件位置列表
        with open(list_path, "r") as f:
            self.img_files = [line for line in f if line.strip()]
            
        # 获得对应的label
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        
        # 目标图片大小
        self.img_shape = (img_size, img_size)
        # 每张图片最大包含box数量
        self.max_objects = 50
    
    def __getitem__(self, index):
        """图片"""
        # 获取index对应的图片，且跳过不是三通道的图片（顺序）
        img = None
        img_path = None
        while (img is None or len(img.shape) != 3):
            img_path = self.img_files[index % len(self.img_files)].strip()
            img = np.array(Image.open(img_path))
            index += 1
            
        """
        填充图像为正方形
        原因：
        1. 锚框都为正方形
        2. 图片无变形/丢失信息
        """
        h, w, _ = img.shape
        diff = np.abs(h - w)
        pad1, pad2 = diff // 2, diff - diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, "constant", constant_values=128) / 255.0
        
        padded_h, padded_w, _ = input_img.shape
        
        # 缩放图片
        input_img = resize(input_img, (*self.img_shape, 3), mode="reflect")
        input_img = np.transpose(input_img, (2, 0, 1))
        
        input_img = torch.from_numpy(input_img).float()
        
        """label"""
        label_path = self.label_files[index % len(self.img_files)].strip()
        
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            
            # 根据 padding 的大小, 更新坐标
            x1 += pad[1][0]
            x2 += pad[1][0]
            y1 += pad[0][0]
            y2 += pad[0][0]
            
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            
            filled_labels = np.zeros((self.max_objects, 5))
            if labels:
                filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
            
            filled_labels = torch.from_numpy(filled_labels)
        
        return img_path, input_img, filled_labels
    
    def __len__(self):
        return len(self.img_files)
    
    
class DetectImgDatasets(Dataset):
    def __init__(self, img_folder, img_size=416):
        # 获取文件夹下所有的图片路径
        self.img_files = glob.glob(f"{img_folder}/*.*")
        self.img_shape = (img_size, img_size)
        
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]
        img = np.array(Image.open(img_path))
        
        h, w, _ = img.shape
        diff = np.abs(h, w)
        pad1, pad2 = diff // 2, diff - diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, mode="constant", constant_values=127.5) / 225
        
        input_img = resize(input_img, (*self.img_shape, 3), mode="reflect")
        input_img = np.transpose(input_img, (2, 0, 1))
        
        input_img = torch.from_numpy(input_img).float()
        return img_path, input_img
    
    def __len__(self):
        return len(self.img_files)
        