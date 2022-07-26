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
    train_dir = r'D:\code_learning\dataset\102flowers\train' # 后续填充
    valid_dir = r'D:\code_learning\dataset\102flowers\valid' # 后续填充
    # image-net 预训练模型地址
    path_state_dict = r'D:\code_learning\model_path\resnet18-f37072fd.pth' # 后续填充
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

    train_data = FlowerDataset(root_dir=train_dir, transform=transforms_train)
    valid_data = FlowerDataset(root_dir=valid_dir, transform=transforms_valid)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs, num_workers=workers)

    # 2. 模型
    model = resnet18()
    # load pretrain model
    if os.path.exists(path_state_dict):
        pretrain_state_dict = torch.load(path_state_dict, map_location="cpu")
        model.load_state_dict(pretrain_state_dict)
    else:
        print(f'path:{path_state_dict} is not exists.')
    
    # 修改最后一层
    num_ftes = model.fc.in_features
    model.fc = nn.Linear(num_ftes, train_data.cls_num)
    # to device
    model.to(device)

    # 3.损失函数、优化器
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=factor, milestones=milestones) # 学习率设置，参考https://zhuanlan.zhihu.com/p/380795956

    # 迭代训练
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    for epoch in range(max_epoch):

        # train
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(train_loader, model, loss_f, optimizer, scheduler, epoch, device, log_interval, max_epoch)

        # valid
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(valid_loader, model, loss_f, device)

        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". format(epoch + 1, max_epoch, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step() # 学习率更新

        # 模型保存
        if best_acc < acc_valid or epoch == max_epoch-1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == max_epoch - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)