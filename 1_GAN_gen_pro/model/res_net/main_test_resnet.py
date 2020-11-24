

import numpy as np
import six
import torch
import os
from datapre.qmdctdataset import QmdctDataset
import matplotlib.pyplot as plt
import torch.nn as nn
from model.official_resNet import resnet34
# from model.liu_rewrite_Spec_resNet import Spec_resNet

# start
from collections import namedtuple
HParams = namedtuple('HParams',
                     'epochs, batch_size, num_classes, start_lrn_rate, lr_decay, feature_row, feature_col, channels, '
                     'is_training, num_residual_units, weight_decay_rate, BN_decay, optimizer, device')
hps = HParams(epochs=50,
              batch_size=8,  # 'number of samples in each iteration',
              num_classes=2,  # 'binary classfication',
              start_lrn_rate=0.001,  # starting learning rate，Adam默认参数也是这个。
              lr_decay=0.95,  # 'decaying rate of learning rate ',
              feature_row=128,  # 256 --> 128 "lt"
              feature_col=1024,  # 124 --> 1024
              channels=4,  # 'number of initial channel'
              is_training=True,  # "is training or not."
              num_residual_units=5,  # 'number of residual unit in each different residual module',
              weight_decay_rate=0.0002,  # 'decaying rate of weight '
              BN_decay=0.9997,  # 目前没用到 todo
              optimizer='adam',  # "optimizer, 'adam' or 'sgd', default to 'adam'."
              device='cpu'  # 切记 todo
              )


def main():
    d_loss = []
    acc = []
    # initialize discriminator, optimizer
    discriminator = resnet34().to(hps.device)  # 官方实现版本
    # discriminator = Spec_resNet(hps).to(hps.device)  # lt复现版本
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=10, gamma=hps.lr_decay)  # 每隔step_size个epoch,lr衰减一次(lr*gamma)

    # get dataloader
    cover_folder = os.path.join('../', 'qmdct', '80-cover')
    cover_name_list = os.listdir(cover_folder)
    imagedataset = QmdctDataset(cover_folder)
    cover_loader = torch.utils.data.DataLoader(imagedataset, batch_size=hps.batch_size, shuffle=False)  # "lt改
    stego_folder = os.path.join('../', 'qmdct', '80-stego')
    stego_name_list = os.listdir(stego_folder)  # same as cover_name_list
    imagedataset = QmdctDataset(stego_folder)
    stego_loader = torch.utils.data.DataLoader(imagedataset, batch_size=hps.batch_size, shuffle=False)

    num_train = len(cover_loader.dataset)  # len of cover_loader or stego_loader are the same
    if num_train % hps.batch_size == 0:
        steps_in_epoch = num_train // hps.batch_size
    else:
        steps_in_epoch = num_train // hps.batch_size + 1

    # train_on_batch in epoch
    for epoch in range(1, hps.epochs + 1):
        scheduler.step()  # 学习率衰减
        print('\nStarting epoch {}/{}'.format(epoch, hps.epochs))
        print('Batch size = {}\nSteps in epoch = {}'.format(hps.batch_size, steps_in_epoch))
        running_corrects = 0.0
        batch_cover_list = []
        batch_stego_list = []  # load cover and stego
        for step, batch_cover in enumerate(cover_loader):
            batch_cover_list.append(batch_cover)
        for step, batch_stego in enumerate(stego_loader):
            batch_stego_list.append(batch_stego)

        for step in range(len(cover_loader.dataset) // hps.batch_size):
            start = hps.batch_size * step
            end = hps.batch_size * (step + 1)
            with torch.enable_grad():
                # 第一个概率大表示是cover，否则是stego。 即0类别是cover，1类别是stego
                optimizer_discriminator.zero_grad()

                d_target_label_cover = torch.full([hps.batch_size], 0, device=hps.device).long()  # cover目标标签是0
                d_on_cover = discriminator(batch_cover_list[step].to(hps.device))
                CELoss = nn.CrossEntropyLoss()
                d_loss_on_cover = CELoss(d_on_cover, d_target_label_cover)
                d_target_label_stego = torch.full([hps.batch_size], 1, device=hps.device).long()  # stego目标标签是1
                d_on_stego = discriminator(batch_stego_list[step].to(hps.device))  # .detach()
                d_loss_on_stego = CELoss(d_on_stego, d_target_label_stego)

                # 计算acc
                _, pre_c = torch.max(d_on_cover, dim=1)
                _, pre_s = torch.max(d_on_stego, dim=1)
                correct1 = torch.sum(pre_c == d_target_label_cover)
                correct2 = torch.sum(pre_s == d_target_label_stego)
                batch_correct_num = correct1 + correct2
                running_corrects += batch_correct_num
                # 计算loss
                d_total_loss = d_loss_on_cover + d_loss_on_stego
                d_total_loss.backward()
                optimizer_discriminator.step()

            if (step + 1) % 5 == 0 or (step + 1) == steps_in_epoch:
                print("Epoch: {}/{}, traing step:{}/{}, loss：{}".
                      format(epoch, hps.epochs, step + 1, steps_in_epoch, d_total_loss.item()))
                print('-' * 60)

        d_loss.append(d_total_loss.item())
        plt.plot(d_loss, 'k*')  #画d_loss
        plt.pause(0.01)  # 自动刷新

        # 记录每个epoch的acc
        epoch_acc = running_corrects.__float__() / (num_train * 2)
        acc.append(epoch_acc)
        plt.plot(acc, 'k+')  # 画acc
        plt.pause(0.01)  # 自动刷新
        print("Epoch: {}/{}, Acc: {:.4f}".format(epoch, hps.epochs, epoch_acc))

    # plot loss
    eval_indices = range(0, hps.epochs, 1)
    plt.plot(eval_indices, d_loss, 'k+', label='discriminator_loss')
    plt.title('discriminator loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.pause(0.01)  # 自动刷新
    # plt.show()

    # plot acc
    eval_indices = range(0, hps.epochs, 1)
    plt.plot(eval_indices, acc, 'k*', label='accuracy')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend(loc='upper right')
    plt.pause(0.01)  # 自动刷新
    # plt.show()


if __name__ == "__main__":
    main()
