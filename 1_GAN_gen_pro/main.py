import os
import torch
import argparse
import time

import utils
from model.stego_gan import SteGan
import matplotlib.pyplot as plt


from collections import namedtuple
HParams = namedtuple('HParams',
                     'batch_size, num_classes, start_lrn_rate, lr_decay, feature_row, feature_col, channels, '
                     'is_training, num_residual_units, weight_decay_rate, BN_decay, optimizer, device')
hps = HParams(batch_size=4,  # 'number of samples in each iteration',
              num_classes=2,  # 'binary classfication',
              start_lrn_rate=0.001,  # 'starting learning rate',
              lr_decay=0.95,  # 'decaying rate of learning rate ',
              feature_row=128,  # 256 --> 128 "lt"
              feature_col=1024,  # 124 --> 1024
              channels=4,  # 'number of initial channel'
              is_training=True,  # "is training or not."
              num_residual_units=5,  # 'number of residual unit in each different residual module',
              weight_decay_rate=0.0002,  # 'decaying rate of weight '
              BN_decay=0.9997,
              optimizer='adam',  # "optimizer, 'adam' or 'sgd', default to 'adam'."
              device='cpu'
              )


def main():
    d_loss = []
    g_loss = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parse = argparse.ArgumentParser(description='Training of Stego-Gan')
    parse.add_argument('--epochs', '-e', default=300, type=int, help='NUmber of epochs to train the model.') # 500 todo 0
    parse.add_argument('--batch-size', '-bs', required=True, type=int, help='The batch size.')
    parse.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the image is stored.')
    parse.add_argument('--logging-folder', '-l', default=os.path.join('.', 'logging'), type=str, help='The root folder\
                      where data about experiments are stored.')
    args = parse.parse_args()

    train_options = {'epochs': args.epochs, 'batch_size': args.batch_size, 'data_dir': args.data_dir, 'logging_folder':
                     args.logging_folder}

    model = SteGan(device)
    print(model)
    print(train_options)

    this_run_folder = utils.create_logging_folder(args.logging_folder)
    train_loader, validate_loader, train_name_list = utils.getdataloader(train_options)
    num_trainimages = len(train_loader.dataset)
    if num_trainimages % train_options['batch_size'] == 0:
        steps_in_epoch = num_trainimages // train_options['batch_size']
    else:
        steps_in_epoch = num_trainimages // train_options['batch_size'] + 1

    for epoch in range(1, train_options['epochs']+1):
        print('\nStarting epoch {}/{}'.format(epoch, train_options['epochs']))
        print('Batch size = {}\nSteps in epoch = {}'.format(train_options['batch_size'], steps_in_epoch))
        epoch_starttime = time.time()
        model.scheduler_discriminator.step()  # 学习率衰减
        for step, image in enumerate(train_loader):     # image[1][0,1,2,3]是每个batch_size里的4个名字
            image = image.to(device)                    # train_name_list[batch_size*step+i]  i∈（0,···batch_size-1）
            start = train_options['batch_size']*step
            end = train_options['batch_size']*(step+1)
            batch_name_list = train_name_list[start: end]
            losses, train_data_list = model.train_on_batch(image, train_options, batch_name_list)

            if (step+1) % 20 == 0 or (step+1) == steps_in_epoch:
                print("Epoch: {}/{}, traing step:{}/{}".format(epoch, train_options['epochs'], step+1, steps_in_epoch))
                print(losses)
                print('-'*60)

        train_duration = time.time() - epoch_starttime
        print('Epoch {} trainging duration {:.2f} seconds.'.format(epoch, train_duration))
        print('-'*60)

        print('Running validation for epoch {}/{}'.format(epoch, train_options['epochs']))
        log = {}
        for step, image in enumerate(validate_loader):
            image = image.to(device)
            validate_log, val_data_list = model.validate_on_batch(image)
            pro_image = val_data_list[0].to(device)
            stego_image = val_data_list[2].to(device)
            print('Epoch: {}/{}, validation step: {}'.format(epoch, train_options['epochs'], step+1))
            print(validate_log)
            for name in validate_log:
                log[name] = []
            for name, loss in validate_log.items():
                log[name].append(loss)
            if epoch % 1 == 0 and step == 1:
                utils.save_qmdct(pro_image, stego_image, epoch, this_run_folder)

        utils.write_log(log, epoch, this_run_folder)
        if epoch % 5 == 0:
            utils.save_checkpoint(model, epoch, this_run_folder)

        d_loss.append(log['d_loss'])
        g_loss.append(log['g_loss'])
    plot_loss(d_loss, g_loss, train_options)


def plot_loss(d_loss, g_loss, train_options):
    eval_indices = range(0, train_options['epochs'], 1)
    # Plot discriminator and generator loss
    plt.plot(eval_indices, d_loss, 'k-', label='discriminator_loss')
    plt.plot(eval_indices, g_loss, 'r--', label='generator_loss')
    plt.title('discriminator and generator loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
