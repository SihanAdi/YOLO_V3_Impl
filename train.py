import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from .utils.utils import load_classes, weights_init_normal
from .utils.parse_config import parse_model_config, parse_data_config
from .utils.datasets import DetectImgDatasets
from models import Darknet

import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--image_folder", type=str, default="data/samples")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model_config_path", type=int, default="config/yolov3.cfg")
parser.add_argument("--data_config_path", type=str, default="config/coco.data")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights")
parser.add_argument("--class_path", type=str, default="data/coco2014/coco.names")
parser.add_argument("--conf_thres", type=float, default=0.8)
parser.add_argument("--nms_thres", type=float, default=0.4)
parser.add_argument("--n_cpu", type=int, default=0)
parser.add_argument("--img_size", type=int, default=416)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--use_cuda", type=bool, default=True)
opt = parser.parse_args()

logging.info(f"opt: {opt}")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 加载类名
classes = load_classes(opt.class_path)

# 加载数据集相关配置
data_config = parse_data_config(opt.data_config_path)
train_data_path = data_config["train"]

# 获取模型超参数
hyper_params = parse_model_config(opt.model_config_path)[0]
lr = float(hyper_params["learning_rate"])
momentum = float(hyper_params["momentum"])
decay = float(hyper_params["decay"])
burn_in = int(hyper_params["burn_in"])

# 初始化模型
model = Darknet(opt.model_config_path)

# 初始化参数
model.apply(weights_init_normal)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.train()

dataloader = DataLoader(
    dataset=DetectImgDatasets(train_data_path),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu
)

optimizer = optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters())
)

for epoch in range(len(opt.epochs)):
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        imgs = imgs.to(dtype=torch.float32, device=device)
        targets = targets.to(dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        optimizer.step()
        
        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            ), flush=True
        )
        model.seen += imgs.shape[0]
    if epoch % opt.checkpoint_interval == 0:
        checkpoint_path = f"{opt.checkpoint_dir}/yolov3_ckpt_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)

