import os

import numpy as np

import torch
import torch.nn as nn

from model.multimap_generator import Unet
from model.discriminator import DNet
from datapre.qmdctdataset import QmdctDataset
# from model.Discriminator_Spec_resNet import Spec_resNet
from model.official_resNet import resnet18



class SteGan(nn.Module):
    def __init__(self, device):
        super(SteGan, self).__init__()

        self.generator = Unet().to(device)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters())

        self.discriminator = resnet18().to(device)  # "lt"加了L2-weight_decay,eps,lr_decay
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), eps=1e-5, weight_decay=0.0002)
        self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
            self.optimizer_discriminator, step_size=10, gamma=0.95)  # 每隔step_size个epoch,学习率衰减一次(lr*gamma)

        self.CELoss = nn.CrossEntropyLoss()
        self.device = device

    def train_on_batch(self, cover, options, batch_name_list):
        batch_size = cover.shape[0]

        with torch.enable_grad():
            # train the discriminator
            # 第一个概率大表示是cover，否则是stego。 即0类别是cover，1类别是stego
            self.optimizer_discriminator.zero_grad()
            d_target_label_cover = torch.full([batch_size], 0, device=self.device).long() #cover目标标签是0
            d_on_cover = self.discriminator(cover)
            #print(cover)
            d_loss_on_cover = self.CELoss(d_on_cover, d_target_label_cover)

            pro_matrix = self.generator(cover)
            rand_matrix = torch.rand(batch_size, 1, pro_matrix.shape[2], pro_matrix.shape[3]).to(self.device)
            modi_matrix = tanh_simulator(pro_matrix, rand_matrix).to(self.device)
            stego = cover + modi_matrix
            # stego_torch_uint8 = stego.detach().byte().float()       # 这个detach()非常重要,stego是float32类型,转化为uint8
            # print(stego_torch_uint8)
            d_target_label_stego = torch.full([batch_size], 1, device=self.device).long() #stego目标标签是1
            # d_on_stego = self.discriminator(stego_torch_uint8)
            d_on_stego = self.discriminator(stego.detach())              # detach不能没有
            #print(stego.detach().dtype)
            d_loss_on_stego = self.CELoss(d_on_stego, d_target_label_stego)

            d_total_loss = d_loss_on_cover + d_loss_on_stego
            d_total_loss.backward()
            self.optimizer_discriminator.step()

            # train the generator
            self.optimizer_generator.zero_grad()
            g_target_label = torch.full([batch_size], 0, device=self.device).long() #因为G要骗过D，这里目标标签是0,即cover的类别。
            # print(stego)
            # stego_grad_torch_uint8 = stego.byte().float()          # 转换类型操作使得参数不更新
            d_on_g_stego = self.discriminator(stego)
            g_loss_adv = self.CELoss(d_on_g_stego, g_target_label)
            g_loss_cpa = capacity_loss(pro_matrix, 0.4)
            g_loss = g_loss_adv + 1e-10 * g_loss_cpa     #  todo 1.1
            g_loss.backward()
            self.optimizer_generator.step()

            losses = {
                'd_loss': d_total_loss.item(),
                'g_loss': g_loss.item()
            }

            # 保存生成的stego  --"lt"
            stego_folder = os.path.join(options['data_dir'], 'train_stego')
            for i in range(stego.shape[0]):
                stego_qmdct = stego.detach()[i].squeeze().data.cpu().numpy()
                np.savetxt(os.path.join(stego_folder, batch_name_list[i]), stego_qmdct, fmt='%d')

            return losses, (pro_matrix, modi_matrix, stego)


    def validate_on_batch(self, cover):
        batch_size = cover.shape[0]
        with torch.no_grad():   # validate 这里模型参数不变 所以是no_grad
            d_target_label_cover = torch.full([batch_size], 0, device=self.device).long()
            d_on_cover = self.discriminator(cover)
            d_loss_on_cover = self.CELoss(d_on_cover, d_target_label_cover)

            pro_matrix = self.generator(cover)
            # print(pro_matrix)
            rand_matrix = torch.rand(batch_size, 1, pro_matrix.shape[2], pro_matrix.shape[3]).to(self.device)
            modi_matrix = tanh_simulator(pro_matrix, rand_matrix).to(self.device)
            # print(modi_matrix)
            stego = cover + modi_matrix
            d_target_label_stego = torch.full([batch_size], 1, device=self.device).long()
            d_on_stego = self.discriminator(stego)
            d_loss_on_stego = self.CELoss(d_on_stego, d_target_label_stego)
            d_total_loss = d_loss_on_cover + d_loss_on_stego

            g_target_label = torch.full([batch_size], 0, device=self.device).long()
            d_on_g_stego = self.discriminator(stego)
            g_loss_adv = self.CELoss(d_on_g_stego, g_target_label)
            g_loss_cpa = capacity_loss(pro_matrix, 0.4)
            g_loss = g_loss_adv + 1e-10 * g_loss_cpa # todo 1.2

            # 计算生成的stego有几个被判断是cover（期待目标）或者stego
            # 目前情况是，生成的stego依然被判断为stego  ---9月4
            _, pre_index = torch.max(d_on_g_stego, dim=1)
            # print(pre_index.shape)
            print(d_on_g_stego)
            print(pre_index)
            num_prestego = torch.sum(pre_index)
            num_precover = batch_size - num_prestego

            validate_logging = {
                'd_loss': d_total_loss.item(),
                'g_loss': g_loss.item(),
                'num_prestego': num_prestego.item(),
                'num_precover': num_precover.item()
            }

            return validate_logging, (pro_matrix, modi_matrix, stego)


def tanh_simulator(p, x):
    y1 = -0.5 * torch.tanh(1000 * (p - 2 * x))
    y2 = 0.5 * torch.tanh(1000 * (p - 2 * (1 - x)))

    return y1 + y2


def capacity_loss(pro, payload):
    pro = pro.squeeze() + 1e-10
    batch_size = pro.shape[0]
    hight = pro.shape[1]
    width = pro.shape[2]
    capa_1 = torch.sum(-pro * torch.log2(pro / 2.0), dim=1)
    capa_2 = torch.sum(-(1-pro) * torch.log2(1 - pro), dim=1)
    capa_1 = torch.sum(capa_1, dim=1)      # batch_size
    #print(capa_1.shape)
    capa_2 = torch.sum(capa_2, dim=1)      # batch_size
    capacity = capa_1 + capa_2
    target = hight * width * payload
    capa_loss = torch.mean((capacity - target).pow(2))

    return capa_loss


def main():
    model = SteGan('cpu')
    print(model)
    train_folder = r'qmdct\train'
    imagedataset = QmdctDataset(train_folder)
    train_loader = torch.utils.data.DataLoader(imagedataset, batch_size=4)
    print(len(train_loader.dataset))
    validate_folder = r'qmdct\val'
    imagedataset = QmdctDataset(validate_folder)
    validate_loader = torch.utils.data.DataLoader(imagedataset, batch_size=4)
    for index, image in enumerate(train_loader):
        print(index)
        #print(image)
        print(image.shape)
        # image_norm = image/255
        loss, train_data = model.train_on_batch(image)
        print(loss)
        #print(stego)
        #print(stego)

    for index, image in enumerate(validate_loader):
        val_logging, val_data = model.validate_on_batch(image)
        print(val_logging)


if __name__ == '__main__':
    main()