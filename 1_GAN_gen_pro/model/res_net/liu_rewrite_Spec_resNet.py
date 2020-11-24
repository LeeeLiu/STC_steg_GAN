import numpy as np
import six

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

import os
from datapre.qmdctdataset import QmdctDataset
import matplotlib.pyplot as plt


class BlockResidual(nn.Module):
    def __init__(self, hps, in_filter, out_filter, stride, activate_before_residual=False):
        super(BlockResidual, self).__init__()
        self.hps = hps
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.stride = stride
        self.activate_before_residual = activate_before_residual
        self.relu = nn.ReLU()
        self._conv1 = nn.Conv2d(in_filter, out_filter, 3, stride, padding=1)
        self._conv2 = nn.Conv2d(out_filter, out_filter, 3, [1, 1], padding=1)  # 原来是[1, 1, 1, 1]
        self.avg_pool = nn.AvgPool2d(self.stride, self.stride)

    def forward(self, x):
        if self.activate_before_residual:
            BN = nn.BatchNorm2d(x.shape[1]).to(self.hps.device)
            x = BN(x)
            x = self.relu(x)
            orig_x = x
        else:
            orig_x = x
            BN = nn.BatchNorm2d(x.shape[1]).to(self.hps.device)
            x = BN(x)
            x = self.relu(x)
        # with tf.variable_scope('sub1'):
        x = self._conv1(x)
        # with tf.variable_scope('sub2'):
        BN = nn.BatchNorm2d(x.shape[1]).to(self.hps.device)
        x = BN(x)
        x = self.relu(x)
        x = self._conv2(x)
        # with tf.variable_scope('sub_add'):
        if self.in_filter != self.out_filter:
            orig_x = self.avg_pool(orig_x)
            orig_x = nn.functional.pad(orig_x, (0, 0, 0, 0,
                                                (self.out_filter - self.in_filter) // 2,
                                                (self.out_filter - self.in_filter) // 2, 0, 0),
                                       "constant")
        x += orig_x
        # print('image after unit %s', x.shape)
        return x


class Spec_resNet(nn.Module):
    def __init__(self, hps):
        super(Spec_resNet, self).__init__()
        # 其他参数
        self.hps = hps
        self.strides = [1, 2, 2]
        self.activate_before_residual = [True, False, False]
        self.filters = [10, 10, 20, 40]
        # 一些网络层
        self.conv = nn.Conv2d(self.hps.channels, 10, 3, [1, 1], padding=1)
        self.layer_A = BlockResidual(hps, self.filters[0], self.filters[1], self.strides[0],
                                     self.activate_before_residual[0])
        self.sublayer_A = self.make_layer(hps, self.filters[1], self.filters[1], [1, 1], False)
        self.layer_B = BlockResidual(hps, self.filters[1], self.filters[2], self.strides[1],
                                     self.activate_before_residual[1])
        self.sublayer_B = self.make_layer(hps, self.filters[2], self.filters[2], [1, 1], False)
        self.layer_C = BlockResidual(hps, self.filters[2], self.filters[3], self.strides[2],
                                     self.activate_before_residual[2])
        self.sublayer_C = self.make_layer(hps, self.filters[3], self.filters[3], [1, 1], False)
        self.BN_ATV = nn.Sequential(
            nn.BatchNorm2d(self.filters[3]),
            nn.ReLU()
        )
        self.layer_final = nn.Sequential(
            nn.Linear(self.filters[3], self.hps.num_classes),
            nn.Softmax(dim=1)
        )

    def make_layer(self, hps, in_filter, out_filter, stride, activate_before_residual):
        layers = []
        for i in six.moves.range(1, self.hps.num_residual_units):
            layers.append(BlockResidual(hps, in_filter, out_filter, stride, activate_before_residual))
        return nn.Sequential(*layers)

    def fixed_conv(self, x, filter_size, in_filters, out_filters, strides, hps):
        """
            Args:
                x:       'batch_size * feature_row * feature_col * in_filters'.
                return:  'batch_size * feature_row * feature_col * out_filters'.
        """
        kernel = np.asarray([0, -1, 0, 0, 1, 0, 0, 0, 0,
                             0, 1, 0, 0, -2, 0, 0, 1, 0,
                             0, 0, 0, -1, 1, 0, 0, 0, 0,
                             0, 0, 0, 1, -2, 1, 0, 0, 0])
        """
        ([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -2, 1], [0, 0, 0]]])                
         """
        kernel = torch.from_numpy(kernel)
        kernel = kernel.float().to(hps.device)
        filters = kernel.view(out_filters, in_filters, filter_size, filter_size)
        return nn.functional.conv2d(x, filters, stride=strides, padding=(filter_size - 1) // 2)

    def forward(self, x):
        x = x.float()  # spec_res = x.view(-1, self.hps.feature_row, self.hps.feature_col, 1)
        # fixed_conv
        spec_filtered = self.fixed_conv(x, 3, 1, self.hps.channels, [1, 1], self.hps)  # 原来是[1, 1, 1, 1]
        # 'init_conv'
        x = self.conv(spec_filtered)
        # conv-A
        x = self.layer_A(x)
        x = self.sublayer_A(x)
        x = self.layer_B(x)
        x = self.sublayer_B(x)
        x = self.layer_C(x)
        x = self.sublayer_C(x)
        x = self.BN_ATV(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.layer_final(x)

        return x


from collections import namedtuple

HParams = namedtuple('HParams',
                     'epochs, batch_size, num_classes, start_lrn_rate, decay_rate, feature_row, feature_col, channels, '
                     'is_training, num_residual_units, weight_decay_rate, BN_decay, optimizer, device')
hps = HParams(epochs=200,
              batch_size=8,  # 'number of samples in each iteration',
              num_classes=2,  # 'binary classfication',
              start_lrn_rate=0.001,  # 'starting learning rate',
              decay_rate=0.95,  # 'decaying rate of learning rate ',
              feature_row=128,  # 256 --> 128 "lt"
              feature_col=1024,  # 124 --> 1024
              channels=4,  # 'number of initial channel'
              is_training=True,  # "is training or not."
              num_residual_units=5,  # 'number of residual unit in each different residual module',
              weight_decay_rate=0.0002,  # 'decaying rate of weight '
              BN_decay=0.9997,
              optimizer='adam',  # "optimizer, 'adam' or 'sgd', default to 'adam'."
              device='cuda'
              )


def main():
    d_loss = []

    # initialize discriminator, optimizer
    discriminator = Spec_resNet(hps).to(hps.device)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), eps=1e-8, weight_decay=0.0002)

    # get dataloader
    cover_folder = os.path.join('../', 'qmdct', 'train')
    cover_name_list = os.listdir(cover_folder)
    imagedataset = QmdctDataset(cover_folder)
    cover_loader = torch.utils.data.DataLoader(imagedataset, batch_size=hps.batch_size, shuffle=False)  # "lt改
    stego_folder = os.path.join('../', 'qmdct', 'train_stego')
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
        print('\nStarting epoch {}/{}'.format(epoch, hps.epochs))
        print('Batch size = {}\nSteps in epoch = {}'.format(hps.batch_size, steps_in_epoch))
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

                d_total_loss = d_loss_on_cover + d_loss_on_stego
                d_total_loss.backward()
                optimizer_discriminator.step()

            if (step + 1) % 10 == 0 or (step + 1) == steps_in_epoch:
                print("Epoch: {}/{}, traing step:{}/{}".format(epoch, hps.epochs, step + 1, steps_in_epoch))
                print(d_total_loss.item())
                print('-' * 60)

        d_loss.append(d_total_loss.item())

    # plot
    eval_indices = range(0, hps.epochs, 1)
    plt.plot(eval_indices, d_loss, 'k-', label='discriminator_loss')
    plt.title('discriminator loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
