#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   parse_error_imgs.py
@Time    :   2022/07/29 19:04:35
@Author  :   Alex Wong 
@Version :   1.0
@Desc    :   将错误分类的图片挑出来，进行观察
'''

import os
import pickle
import shutil


def load_pickle(path_file):
    with open(path_file, "rb") as f:
        data = pickle.load(f)
    return data

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):  # os.path.isdir() 判断对象是否为一个目录
        os.makedirs(my_dir)

def main():
    path_pkl = r"path" # 错误图片数据集路径
    data_root_dir = r"path" # 图像路径
    out_dir = path_pkl[:-4] # 输出文件目录
    error_info = load_pickle(path_pkl)

    for setname, info in error_info.items():
        for imgs_data in info:
            label, pred, path_img_rel = imgs_data
            path_img = os.path.join(data_root_dir, os.path.basename(path_img_rel)) # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。 参考https://blog.csdn.net/qq_45893319/article/details/123424519
            img_dir = os.path.join(out_dir, setname, str(label), str(pred))
            my_mkdir(img_dir)
            shutil.copy(path_img, img_dir)

if __name__ == '__main__':
    main()