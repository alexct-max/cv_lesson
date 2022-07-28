#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   evaluate_flower.py
@Time    :   2022/07/28 16:50:59
@Author  :   Alex Wong 
@Version :   1.0
@Desc    :   模型在test上进行指标计算
'''


import torch
import numpy as np
import torch.nn as nn
from datasets.flower_102 import FlowerDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # config
    data_dir = r"path"  # 测试集地址
    path_state_dict = r"path"  # 模型保存地址

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform
    ])
    test_bs = 64
    workers = 4
    
    test_data = FlowerDataset(root_dir=data_dir, transform=transforms_test)
    test_loader = DataLoader(dataset=test_data, batch_size=test_bs, num_workers=workers)

    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.cls_num)
    # load pretrain model


if __name__ == '__main__':
    main()