import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datapre.qmdctdataset import QmdctDataset


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        kernel = (1/12)*np.asarray([[-1, 2, -2, 2, -1],
                                    [2, -6, 8, -6, 2],
                                    [-2, 8, -12, 8, -2],
                                    [2, -6, 8, -6, 2],
                                    [-1, 2, -2, 2, -1]])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.hpf = nn.Parameter(data=kernel, requires_grad=False)

        blocks = []
        blocks.append(BlockType1(1, 8))         # 64*512
        blocks.append(BlockType1(8, 16))        # 32*256
        blocks.append(BlockType1(16, 32))       # 16*128
        blocks.append(BlockType1(32, 64))       # 8*64
        blocks.append(BlockType1(64, 128))      # 4*32
        blocks.append(BlockType1(128, 256))     # 2*16
        self.layers = nn.Sequential(*blocks)

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(2, 2))       # 256*2*2
        self.fullcon1 = nn.Linear(1024, 256)
        self.fullcon2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.hpf, padding=2)
        x2 = self.layers(x1)
        x3 = self.maxpool(x2)
        x3_flat = x3.view(x3.shape[0], 1024)  # torch.Size([4, 1024])
        x4 = self.fullcon1(x3_flat) # torch.Size([4, 256])
        x5 = self.fullcon2(x4) # torch.Size([4, 2])
        out = self.softmax(x5)

        return out

class BlockType1(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(BlockType1, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels_out),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class BlockType4(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(BlockType4, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=32)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


def main():
    model = DNet()
    x = torch.rand(4, 1, 128, 1024)
    y = model(x)
    # print(y)
    # print(y.shape)

    # model = XuNet()
    # train_folder = r'D:\Programes\Python Examples\Gan-STC\image\train'
    # imagedataset = BossBaseDataset(train_folder)
    # train_loader = torch.utils.data.DataLoader(imagedataset, batch_size=4)
    # for index, image in enumerate(train_loader):
    #     print(index)
    #     print(image)
    #     print(image.shape)
    #     image_norm = image/255
    #     pre = model(image_norm)
    #     print(pre.shape)
    #     print(pre)


if __name__ == "__main__":
    main()