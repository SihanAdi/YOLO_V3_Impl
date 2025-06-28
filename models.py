"""
YOLO模型架构
"""

import torch.nn as nn
from .utils.parse_config import parse_model_config

def create_modules(module_defs):
    pass


class EmptyLayer(nn.Module):
    """起到占位符(placeholder)的作用"""
    def __init__(self) -> None:
        super().__init__()
        

