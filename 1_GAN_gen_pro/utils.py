import os
import time

import csv
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from datapre.qmdctdataset import QmdctDataset


def create_logging_folder(logging_folder):
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)

    this_run_time = time.strftime('%Y.%m.%d--%H-%M-%S')
    this_run_foler = os.path.join(logging_folder, this_run_time)
    os.makedirs(this_run_foler)
    os.makedirs(os.path.join(this_run_foler, 'checkpoints'))
    os.makedirs(os.path.join(this_run_foler, 'proimages'))
    os.makedirs(os.path.join(this_run_foler, 'stegoimages'))

    return this_run_foler


def getdataloader(options):
    train_folder = os.path.join(options['data_dir'], 'train')
    train_name_list = os.listdir(train_folder)      # 记录所有名字
    imagedataset = QmdctDataset(train_folder)
    # _, name1 = imagedataset.__getitem__(3)
    train_loader = torch.utils.data.DataLoader(imagedataset, batch_size=options['batch_size'], shuffle=False)  # "lt改
    # _, name2 = train_loader.dataset.__getitem__(3)
    val_folder = os.path.join(options['data_dir'], 'val')
    imagedataset = QmdctDataset(val_folder)
    val_loader = torch.utils.data.DataLoader(imagedataset, batch_size=options['batch_size'], shuffle=False)

    return train_loader, val_loader, train_name_list


def save_image(pro, stego, epoch, folder):
    num_images = pro.shape[0]
    stego_image_array = stego.squeeze().numpy().round().astype('uint8')
    pro_enlarge = pro * 255 * (10.0/5.0)
    print(pro_enlarge)
    pro_array = pro_enlarge.squeeze().numpy().round().astype('uint8')

    for i in range(num_images):
        im_stego = Image.fromarray(stego_image_array[i])
        im_pro = Image.fromarray(pro_array[i])
        stego_folder = os.path.join(folder, 'stegoimages')
        pro_folder = os.path.join(folder, 'proimages')
        im_stego.save(os.path.join(stego_folder, 'stego_{}_{}.png'.format(epoch, i)))
        im_pro.save(os.path.join(pro_folder, 'pro_{}_{}.png'.format(epoch, i)))


def save_qmdct(pro, stego, epoch, folder):
    num_qmdct = pro.shape[0]
    stego_qmdct_array = stego.squeeze().data.cpu().numpy().round().astype('int32')
    pro_enlarge = pro * 255 * (10.0 / 5.0) # todo 3
    print(pro_enlarge)
    pro_array = pro_enlarge.squeeze().data.cpu().numpy().round().astype('uint8')

    for i in range(num_qmdct):
        stego_qmdct = stego_qmdct_array[i]
        im_pro = Image.fromarray(pro_array[i])
        stego_folder = os.path.join(folder, 'stegoimages')
        pro_folder = os.path.join(folder, 'proimages')
        im_pro.save(os.path.join(pro_folder, 'pro_{}_{}.png'.format(epoch, i))) # 把概率图pro放大至0-255范围 保存为图像
        np.savetxt(os.path.join(stego_folder, 'stego_{}_{}.txt'.format(epoch, i)), stego_qmdct, fmt='%d')


def write_log(log, epoch, folder):
    file_name = os.path.join(folder, 'validation.csv')
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_header = ['epoch'] + [loss_name for loss_name in log.keys()]
            writer.writerow(row_to_header)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in log.values()]
        writer.writerow(row_to_write)


def save_checkpoint(model, epoch, folder):
    checkpoint_folder = os.path.join(folder, 'checkpoints')
    checkpoint_name = 'checkpoint_of_epoch_{}.pyt'.format(epoch)
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_name)
    print('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'epoch': epoch,
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict()
    }
    torch.save(checkpoint, checkpoint_filename)
    print('Saving checkpoint done!')


def main():
    log = {
        'd_loss': [1.265, 1.234],
        'g_loss': [1.786, 1.864],
        'num_precover': [9, 8],
        'num_prestego': [7, 10]
    }
    print([names for names in log.keys()])
    print([values for values in log.values()])
    print([np.mean(values) for values in log.values()])
    for i in range(5):
        write_log(log, 1, 'logging')


if __name__ == "__main__":
    main()

