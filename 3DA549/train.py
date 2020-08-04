# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import argparse
from tensorboardX import *
from torch.utils.data import DataLoader
import datetime
from tqdm import *
import os
import cv2

from dataset import litDataset
from lit.model import unet
from lit.loss import dice_loss
from lit.metrics import get_ious


def main(args):
    # tensorboard
    temp = os.path.join(args.path_to_log, 'test_{}'.format(datetime.datetime.now().strftime('%m_%d %H %M')))
    if not os.path.isdir(temp):
        os.mkdir(temp)
    logger_path = temp
    logger_tb = SummaryWriter(log_dir=logger_path)

    # create dataset
    train_dataset = litDataset(args.train_data)

    # create data loader
    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_dataset, **train_params)

    # model
    model = unet.UNet3D(args.num_kernel, args.kernel_size, train_dataset.dim, train_dataset.target_dim)

    # device
    device = torch.device(args.device)
    if args.device == "cuda":
        # parse gpu_ids for data parallel
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)
        # parallelize computation
        if type(gpu_ids) is not int:
            model = nn.DataParallel(model, gpu_ids)
    model.to(device)

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss
    loss_function = dice_loss

    # train model
    for epoch in range(args.epoch):
        model.train()
        out = '|| Epoch ' + str(epoch) + '/' + str(args.epoch-1) + ' ||'
        print('=' * len(out))
        print(out)
        print('=' * len(out))

        total_loss = []
        total_iou = []
        for i, (x_train, y_train) in tqdm(enumerate(train_dataloader)):
            with torch.set_grad_enabled(True):
                # send data to device
                x = torch.Tensor(x_train.float()).to(device)
                y = torch.Tensor(y_train.float()).to(device)
                # predict segmentation
                pred = model.forward(x)

                # calculate loss
                # print(pred.shape, y.shape)
                loss = loss_function(pred, y)
                total_loss.append(loss.item())
                # calculate IoU
                # to numpy array
                # squeeze: 压缩掉维度为1的维度
                # detach: 返回无gradient的tensor
                predictions = pred.clone().squeeze().detach().cpu().numpy()
                gt = y.clone().squeeze().detach().cpu().numpy()

                ious = [get_ious(p, g, 0.5) for p, g in zip(predictions, gt)]
                total_iou.append(np.mean(ious))
                # print('Step: {:02d} | Train_loss: {:.4f} | IOU: {:.2f}'.format(i, loss.item(), np.mean(ious)))

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i == 0:
                # display segmentation on tensorboard
                # print(x_train[0].shape, y_train[0].shape, pred[0].shape)
                original = x_train[0][:, 30, :, :]
                truth = y_train[0][:, 30, :, :].squeeze()
                seg = pred[0][:, 30, :, :].cpu().squeeze().detach().numpy()

                logger_tb.add_image("truth", truth, epoch)
                logger_tb.add_image("segmentation", seg, epoch)
                logger_tb.add_image("original", original, epoch)

        # mean loss and iou
        avg_loss = np.mean(total_loss)
        avg_iou = np.mean(total_iou)

        logger_tb.add_scalar("train loss", avg_loss, epoch)
        logger_tb.add_scalar("train iou", avg_iou, epoch)

        print('Epoch {:02d} | Train_loss: {:.4f} | IOU: {:.2f}'.format(epoch, np.mean(total_loss), np.mean(total_iou)))

    # save model
    model_name = 'model_PREDICT_{}.h5'.format(datetime.datetime.now().strftime('%m%d_%H_%M'))
    ckpt_path = os.path.join(args.path_to_model, model_name)
    ckpt_dict = model.state_dict()
    torch.save(ckpt_dict, ckpt_path)

    logger_tb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        default="D:\\LitData\\Fluo-C3DH-A549\\TRAIN",
                        help='path of dataset')
    parser.add_argument('--path_to_model', type=str,
                        default='D:\\LitData\\Fluo-C3DH-A549\\models',
                        help='path to save model parameters')
    parser.add_argument('--path_to_log', type=str,
                        default='D:\\LitData\\Fluo-C3DH-A549\\logs',
                        help='path to log training process')
    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='1')
    args = parser.parse_args()

    main(args)