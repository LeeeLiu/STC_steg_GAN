
import torch
import torch.nn as nn
from torch.nn import functional as F

from datapre.qmdctdataset import QmdctDataset


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.layer1 = UnetDownConv(1, 16)      # 64*512
        self.layer2 = UnetDownConv(16*3, 32)     # 32*256
        self.layer3 = UnetDownConv(32*3, 64)     # 16*128
        self.layer4 = UnetDownConv(64*3, 128)    # 8*64
        self.layer5 = UnetDownConv(128*3, 128)   # 4*32
        self.layer6 = UnetDownConv(128*3, 128)   # 2*16
        self.layer7 = UnetUpConv(128*3, 128)      # 4*32
        self.layer8 = UnetUpConv((128*4), 128)     # 8*64
        self.layer9 = UnetUpConv((128*4), 64)     # 16*128
        self.layer10 = UnetUpConv((64*4), 32)     # 32*256
        self.layer11 = UnetUpConv((32*4), 16)      # 64*512
        self.layer12 = UnetUpConv((16*4), 1)      # 128*1024
        self.act_fun1 = nn.Sigmoid()            # -0.5 -- 0.5
        self.act_fun2 = nn.ReLU()               # 0 -- 0.5

    def forward(self, x):
        # print(x.shape)
        x1 = self.layer1(x)
        # print(x1.shape)
        x2 = self.layer2(x1)
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)
        # print(x4.shape)
        x5 = self.layer5(x4)
        # print(x5.shape)
        x6 = self.layer6(x5)
        # print(x6.shape)
        x7 = self.layer7(x6, x5)
        # print(x7.shape)
        x8 = self.layer8(x7, x4)
        # print(x8.shape)
        x9 = self.layer9(x8, x3)
        # print(x9.shape)
        x10 = self.layer10(x9, x2)
        # print(x10.shape)
        x11 = self.layer11(x10, x1)
        # print(x11.shape)
        x12 = self.layer12(x11)
        # print(x12.shape)
        x13 = self.act_fun1(x12) - 0.5     # -0.5 -- 0.5 todo 2
        # print(x13.shape)
        x14 = self.act_fun2(x13)           # 0 -- 0.5

        return x14

class UnetDownConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UnetDownConv, self).__init__()
        #  padding = (f-1)/2
        self.layers1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        feature_map1 = self.layers1(x)
        feature_map2 = self.layers2(x)
        feature_map3 = self.layers3(x)
        combine_fm = torch.cat([feature_map1, feature_map2, feature_map3], dim=1)
        return combine_fm

class UnetUpConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UnetUpConv, self).__init__()

        self.layers = nn.Sequential(
            # 把反卷积换成最近邻插值+正常卷积
            # p=(f-1)/2, stride=1
            nn.Conv2d(channels_in, channels_out, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y=torch.Tensor(0)):
        x = F.interpolate(x, size=[2*x.shape[2],2*x.shape[3]], scale_factor=None, mode='nearest', align_corners=None)
        x_tmp = self.layers(x)
        out = torch.cat([x_tmp, y], dim=1)      # skip connections
        return out

# class UnetUpConv(nn.Module):
#     def __init__(self, channels_in, channels_out):
#         super(UnetUpConv, self).__init__()
#         #  output_padding = stride-1，  padding = (kernel_size - 1)/2
#         self.layers = nn.Sequential(
#             nn.ConvTranspose2d(channels_in, channels_out, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.BatchNorm2d(channels_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x, y=torch.Tensor(0)):
#         x_tmp = self.layers(x)
#         out = torch.cat([x_tmp, y], dim=1)     # skip connections
#         return out

def main():
    model = Unet()
    input = torch.rand(2, 1, 128, 1024)     # lt
    print(input.shape)
    print("\n")
    output = model(input)
    print("\n")
    print(output.shape)

if __name__ == "__main__":
    main()