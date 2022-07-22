# -*- encoding: utf-8 -*-
'''
@File    :   model_trainer.py
@Time    :   2022/07/22 22:49:45
@Author  :   Wang Wei 
@Version :   1.0
@Contact :   wow.kingofmoon@gmail.com
@github  :   https://github.com/alexct-max/cv_lesson.git
@Desc    :   None
'''

# here put the import lib
import torch
import numpy as np
from collections import Counter


class ModelTrainer(object):
    # @staticmethod 参考https://github.com/taizilongxu/interview_python#3-staticmethod%E5%92%8Cclassmethod
    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, log_interval, max_epoch):
        model.train()
        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = []
        acc_avg = []
        path_error = []
        label_list = []
