# -*- encoding: utf-8 -*-
'''
@File    :   flowe_train.py
@Time    :   2022/07/22 22:38:31
@Author  :   Wang Wei 
@Version :   1.0
@Contact :   wow.kingofmoon@gmail.com
@github  :   https://github.com/alexct-max/cv_lesson.git
@Desc    :   训练模型
'''

# here put the import lib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.transforms as transforms
from datasets.flower_102 import FlowerDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tools.model_trainer import ModelTrainer
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 0. config
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%Y-%m-%d_%H_%M")
    log_dir = os.path.join(BASE_DIR, '..', '..', 'results', time_str) # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):  # 如果path存在，返回True；如果path不存在，返回False。 参考https://blog.csdn.net/weixin_41093268/article/details/82735035
        os.makedirs(log_dir)
    # 训练集、测试集
    train_dir = r'path' # 后续填充
    valid_dir = r'path' # 后续填充
    # image-net 预训练模型地址
    path_state_dict = r'path' # 后续填充
    # batch size
    train_bs = 64
    valid_bs = 64
    workers = 0 # 多线程读取数据
    # 损失函数、优化器相关参数
    lr_init = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    factor = 0.1
    milestones = [30, 45]
    max_epoch = 50

    log_interval = 10 # log 间隔

    # 1. 数据
    norm_mean = [0.485, 0.456, 0.406] # imagenet 120万图像统计得来
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_train = transforms.Compose([
        transforms.Resize((256)),  # (256, 256) 区别； （256） 最短边256
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5), # 随机翻转
        transforms.ToTensor(),
        normTransform
    ]) # transforms 常用参数，参考http://t.zoukankan.com/zhangxiann-p-13570884.html
    transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)), # 图像失真，可能影响效果
        transforms.ToTensor(),
        normTransform
    ])
