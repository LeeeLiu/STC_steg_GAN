import os
import argparse

import torch

import utils
from model import stego_gan
from model.stego_gan import SteGan
from datapre.qmdctdataset import QmdctDataset
#from datapre.qmdctdataset import BossBaseDataset


def model_from_checkpoint(model, checkpoint):
    model.generator.load_state_dict(checkpoint['generator'])
    model.discriminator.load_state_dict(checkpoint['discriminator'])


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # test_model.py -ckp logging/2019.09.03--22-38-24/checkpoints/checkpoint_of_epoch_5.pyt -tf qmdct/test ---lt标注 9.4
    parser = argparse.ArgumentParser(description='Testing of SteGan')
    parser.add_argument('--checkpoint-path', '-ckp', required=True, type=str)
    parser.add_argument('--test-folder', '-tf', required=True, type=str)
    parser.add_argument('--result-folder', '-rf', default='', type=str)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    test_path = args.test_folder
    result_path = args.result_folder
    if result_path == '':
        result_path = test_path + '_out_' + checkpoint_path.split('/')[-1]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(os.path.join(result_path, 'proimages'))
        os.makedirs(os.path.join(result_path, 'stegoimages'))

    model = SteGan(device)
    print('=> Loading checkpoint {}.'.format(checkpoint_path))
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('=> Loading checkpoint done(epoch{}).'.format(checkpoint['epoch']))
    model_from_checkpoint(model, checkpoint)

    # testdataset = BossBaseDataset(test_path)  lt修改 ----9.4
    testdataset = QmdctDataset(test_path)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=4, shuffle=False)
    print('Start testing the model.')
    for step, testdata in enumerate(testloader):
        image = testdata.to(device)
        pro_image = model.generator(image).detach()
        rand_matrix = torch.rand(pro_image.shape[0], pro_image.shape[1], pro_image.shape[2], pro_image.shape[3]).to(device)
        modi_image = stego_gan.tanh_simulator(pro_image, rand_matrix)
        stego_image = image + modi_image
        utils.save_image(pro_image.to(device), stego_image.to(device), 'test_{}'.format(step), result_path)
    print('Test the model done.')


if __name__ == '__main__':
    main()