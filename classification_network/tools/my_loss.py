# -*- encoding: utf-8 -*-
'''
@File    :   my_loss.py
@Time    :   2022/07/26 23:25:06
@Author  :   Wang Wei 
@Version :   1.0
@Contact :   wow.kingofmoon@gmail.com
@github  :   https://github.com/alexct-max/cv_lesson.git
@Desc    :   Label Smooth Loss 实现
'''

# here put the import lib
from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        # 0. inputs 模型输出的预测结果；target 标签
        # 1. softmax
        # 2. 制作权重，真实类别的权重为 1-smoothing， 其余类别权重为  (1-smoothing) / (K-1)
        # 3. 依交叉熵损失函数公式计算loss
        # 参考https://zhuanlan.zhihu.com/p/116466239
        log_prob = F.log_softmax(inputs, dim=-1)  # log(pi)
        weight = inputs.new_ones(inputs.size()) * self.smoothing/(inputs.size(-1) - 1.) # new_ones() 返回一个与size大小相同的用1填充的张量。 默认返回的Tensor具有与此张量相同的torch.dtype和torch.device，参考https://blog.csdn.net/Dontla/article/details/104675406/
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing)) # unsqueeze(), 插入维度，参考https://blog.csdn.net/ljwwjl/article/details/115342632；scatter_(input, dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input, 参考https://blog.csdn.net/weixin_45547563/article/details/105311543
        loss = (-weight * log_prob).sum(dim=-1).mean() # -log(pi)*yi
        return loss

def main():

    output = torch.tensor([[4.0, 15.0, 10.0], [11.0, 5.0, 4.0], [1.0, 5.0, 24.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    criterion = LabelSmoothLoss(0.001)
    loss = criterion(output, label)

    print(f"CrossEntropy:{loss}")

if __name__ == '__main__':
    main()