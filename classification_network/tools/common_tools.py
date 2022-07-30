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
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel("Predict label")
    plt.ylabel("True label")
    plt.title(f"Confusion_Matrix_{set_name}_{epoch}")

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va="center", ha="center", color="red", fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(conf_mat_per[i, j]), va="center", ha="center", color="red", fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, f"Confusion_Matrix_{set_name}.png"))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i], confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])), confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i])))) # https://blog.csdn.net/W1995S/article/details/114988637


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')
    plt.ylabel(str(mode))
    plt.xlabel("Epoch")

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log) # os.path.basename() 将指定的路径拆分为后返回尾部， 参考https://vimsky.com/examples/usage/python-os-path-basename-method.html
        self.log_name = log_name if log_name else 'root'
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path) # os.path.dirname() 去掉文件名，返回目录，参考https://blog.csdn.net/weixin_38470851/article/details/80367143
        