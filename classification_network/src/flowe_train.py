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
