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
    # 可以使用静态方法调用，即 ModelTraine.train()调用
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

        for i, data in enumerate(data_loader):

            _, label = data
            # extend()函数：作用是扩展一个列表，这个函数类似把两个列表相加，参考https://www.py.cn/jishu/jichu/20402.html
            label_list.extend(label.tolist())
            # inputs, labels, path_imgs = data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad() # 以SGD为例，是算一个batch计算一次梯度，然后进行一次梯度更新。这里梯度值就是对应偏导数的计算结果。显然，我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0，参考https://blog.csdn.net/bigbigvegetable/article/details/114674793
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward() # 根据loss来计算网络参数的梯度
            optimizer.step() # 针对计算得到的参数梯度对网络参数进行更新，参考https://blog.csdn.net/weixin_45180140/article/details/122047545

            # 统计loss
            loss_sigma.append(loss.item()) #取出 单元素张量 的元素值并返回该值，保持原元素类型不变。一般运用在 神经网络 中用 loss.item()取loss值。参考https://blog.csdn.net/weixin_46568462/article/details/124511177
            loss_mean = np.mean(loss_sigma) 

            _, predicted = torch.max(outputs.data, 1) # 返回输入tensor中所有元素的最大值，参考https://blog.csdn.net/weixin_43635550/article/details/100534904
            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy() #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要将它转化为 CPU tensor,参考https://zhuanlan.zhihu.com/p/165219346
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.