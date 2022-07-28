#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   reorder_flower_dataset.py
@Time    :   2022/07/28 20:23:26
@Author  :   Alex Wong 
@Version :   1.0
@Desc    :   将flower数据集按类别排放，便于分析
'''


import os
import shutil
from scipy.io import loadmat


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def main():
    root_dir = r'D:\code_learning\dataset\102flowers'
    path_mat = r'path' # 标签地址
    reorder_dir = os.path.join(root_dir, "reorder")
    jpg_dir = os.path.join(root_dir, "jpg")

    label_array = loadmat(path_mat)["labels"].squeeze()

    names = os.listdir(jpg_dir)
    names = [p for p in names if p.endswith(".jpg")]
    for name in names:
        idx = int(name[6:11])
        label = label_array[idx-1]-1
        out_dir = os.path.join(reorder_dir, str(label))
        path_src = os.path.join(jpg_dir, name)
        my_mkdir(out_dir)
        shutil.copy(path_src, out_dir)  # 复制文件


if __name__ == "__main__":
    main()