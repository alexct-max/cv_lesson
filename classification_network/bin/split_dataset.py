# -*- encoding: utf-8 -*-
'''
@File    :   split_dataset.py
@Time    :   2022/07/19 23:48:21
@Author  :   Wang Wei 
@Version :   1.0
@Contact :   wow.kingofmoon@gmail.com
@Desc    :   None
'''


import os
import random
import shutil


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):  # os.path.isdir() 判断对象是否为一个目录
        os.makedirs(my_dir)       # os.makedirs() 创建目录


def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for idx, path_img in enumerate(imgs):  # enumerate() 枚举, for循环中返回索引和内容
        if idx % 100 == 0 or idx+1 == len(imgs):
            print("{}/{}".format(idx + 1, len(imgs)))
        '''
        shutil.copy(source, destination)
        shutil.copy() 函数实现文件复制功能，将 source 文件复制到 destination 文件夹中，两个参数都是字符串格式。如果 destination 是一个文件名称，那么它会被用来当作复制后的文件名称，即等于 复制 + 重命名
        '''
        shutil.copy(path_img, data_dir)
    print("{} dataset, copy {} imgs to {}.".format(setname, len(imgs), data_dir))


def main():
    random_seed = 19921225
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    root_dir = r'D:\code_learning\dataset\102flowers'

    data_dir = os.path.join(root_dir, 'jpg')
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    # endswith()是用于判断一个字符串是否以特定的字符串后缀结尾，如果是则返回逻辑值True，否则返回逻辑值False.
    # name_imgs 返回图片列表
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith('.jpg')]
    # path_imgs 返回图片地址列表
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]
    random.seed(random_seed)
    random.shuffle(path_imgs)

    all_nums = len(path_imgs)
    train_breakpoints = int(all_nums*train_ratio)
    valid_breakpoints = int(all_nums*(train_ratio+valid_ratio))
    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    move_img(train_imgs, root_dir, 'train')
    move_img(valid_imgs, root_dir, 'valid')
    move_img(test_imgs, root_dir, 'test')


if __name__ == '__main__':
    main()
