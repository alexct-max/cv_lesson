#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   evaluate_flower.py
@Time    :   2022/07/28 16:50:59
@Author  :   Alex Wong 
@Version :   1.0
@Desc    :   模型在test上进行指标计算
'''

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(BASE_DIR, '..'))
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
    data_dir = r"D:\code_learning\dataset\102flowers\test"  # 测试集地址
    path_state_dict = r"D:\code_learning\cv_lesson\results\2022-07-28_00_38\checkpoint_best.pkl"  # 模型保存地址

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform
    ])
    test_bs = 64
    workers = 0
    
    test_data = FlowerDataset(root_dir=data_dir, transform=transforms_test)
    test_loader = DataLoader(dataset=test_data, batch_size=test_bs, num_workers=workers)

    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.cls_num)
    # load pretrain model
    ckpt = torch.load(path_state_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # step3: inference
    class_num = test_loader.dataset.cls_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        # 统计混淆矩阵
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.

    acc_avg = conf_mat.trace() / conf_mat.sum()
    print("test acc: {:.2%}".format(acc_avg))

if __name__ == '__main__':
    main()