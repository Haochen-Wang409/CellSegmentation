# coding=utf-8

import os
import torch
import torch.nn as nn
import lit.model.layers.unet_layers as layers


class UNet(nn.Module):
    '''
    During the training, the ground truth of each sample is given by two binary masks. After the training,
    the network predicts for each pixel four proximity values of being/not being a marker, and being/not being a cell mask.

    These proximities are not binary. They form four grayscale images.

    one for the foreground
    one for the background
    '''
    def __init__(self, num_kernel, kernel_size, dim, target_dim):
        """UNet

        Parameters
        ----------
            num_kernel: int
                number of kernels to use for the first layer
            kernel_size: int
                size of the kernel for the first layer
            dims: int
                input data dimention
        """

        super(UNet, self).__init__()

        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.dim = dim
        self.target_dim = target_dim

        # encode
        self.encode_1 = layers.DownSampling(self.dim, num_kernel, kernel_size)
        self.encode_2 = layers.DownSampling(num_kernel, num_kernel*2, kernel_size)
        self.encode_3 = layers.DownSampling(num_kernel*2, num_kernel*4, kernel_size)
        self.encode_4 = layers.DownSampling(num_kernel*4, num_kernel*8, kernel_size)

        # bridge
        self.bridge = nn.Conv2d(num_kernel*8, num_kernel*16, kernel_size, padding=1, stride=1)

        # decode
        self.decode_4 = layers.UpSampling(num_kernel*16, num_kernel*8, kernel_size)
        self.decode_3 = layers.UpSampling(num_kernel*8, num_kernel*4, kernel_size)
        self.decode_2 = layers.UpSampling(num_kernel*4, num_kernel*2, kernel_size)
        self.decode_1 = layers.UpSampling(num_kernel*2, num_kernel, kernel_size)

        self.segment = nn.Conv2d(num_kernel, self.target_dim, 1, padding=0, stride=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):

        x, skip_1 = self.encode_1(x)
        x, skip_2 = self.encode_2(x)
        x, skip_3 = self.encode_3(x)
        x, skip_4 = self.encode_4(x)

        x = self.bridge(x)

        x = self.decode_4(x, skip_4)
        x = self.decode_3(x, skip_3)
        x = self.decode_2(x, skip_2)
        x = self.decode_1(x, skip_1)

        x = self.segment(x)

        pred = self.activate(x)

        return pred

    def args_dict(self):
        """model arguments to be saved
        """

        model_args = {'dim': self.dim,
                      'target_dim': self.target_dim,
                      'num_kernel': self.num_kernel,
                      'kernel_size': self.kernel_size}

        return model_args

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)

            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class UNet3D(nn.Module):
    '''
        During the training, the ground truth of each sample is given by two binary masks. After the training,
        the network predicts for each pixel four proximity values of being/not being a marker, and being/not being a cell mask.

        These proximities are not binary. They form four grayscale images.

        one for the foreground
        one for the background
        '''

    def __init__(self, num_kernel, kernel_size, dim, target_dim):
        """UNet

        Parameters
        ----------
            num_kernel: int
                number of kernels to use for the first layer
            kernel_size: int
                size of the kernel for the first layer
            dims: int
                input data dimention
        """

        super(UNet3D, self).__init__()

        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.dim = dim
        self.target_dim = target_dim

        # encode
        self.encode_1 = layers.DownSampling3D(self.dim, num_kernel, kernel_size)
        self.encode_2 = layers.DownSampling3D(num_kernel, num_kernel * 2, kernel_size)
        self.encode_3 = layers.DownSampling3D(num_kernel * 2, num_kernel * 4, kernel_size)
        self.encode_4 = layers.DownSampling3D(num_kernel * 4, num_kernel * 8, kernel_size)

        # bridge
        self.bridge = nn.Conv3d(num_kernel * 8, num_kernel * 16, kernel_size, padding=1, stride=1)

        # decode
        self.decode_4 = layers.UpSampling3D(num_kernel * 16, num_kernel * 8, kernel_size)
        self.decode_3 = layers.UpSampling3D(num_kernel * 8, num_kernel * 4, kernel_size)
        self.decode_2 = layers.UpSampling3D(num_kernel * 4, num_kernel * 2, kernel_size)
        self.decode_1 = layers.UpSampling3D(num_kernel * 2, num_kernel, kernel_size)

        self.segment = nn.Conv3d(num_kernel, self.target_dim, 1, padding=0, stride=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):

        x, skip_1 = self.encode_1(x)
        x, skip_2 = self.encode_2(x)
        x, skip_3 = self.encode_3(x)
        x, skip_4 = self.encode_4(x)

        x = self.bridge(x)

        x = self.decode_4(x, skip_4)
        x = self.decode_3(x, skip_3)
        x = self.decode_2(x, skip_2)
        x = self.decode_1(x, skip_1)

        x = self.segment(x)

        pred = self.activate(x)

        return pred

    def args_dict(self):
        """model arguments to be saved
        """

        model_args = {'dim': self.dim,
                      'target_dim': self.target_dim,
                      'num_kernel': self.num_kernel,
                      'kernel_size': self.kernel_size}

        return model_args

    def init_weights(self, pretrained='', ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)

            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)