#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   common_tools.py
@Time    :   2022/07/29 19:57:55
@Author  :   Alex Wong 
@Version :   1.0
@Desc    :   通用函数库
'''


import os
import logging
from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import torchvision.transforms as transform


def setup_seed(seed=19930315):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)   # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # https://blog.csdn.net/weixin_41990278/article/details/106268969
        torch.backends.cudnn.benckmark = True # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法

def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, figsize=None, perc=False):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >=100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10] # s生成6~30之间的数
    plt.figure(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap("Greys") # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    