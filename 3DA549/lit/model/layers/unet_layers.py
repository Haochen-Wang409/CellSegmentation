import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, name=None):
        super(ConvBlock, self).__init__()

        block = []
        # first conv layer
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, \
                               padding=padding, stride=stride))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        # second conv layer
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size, \
                               padding=padding, stride=stride))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        # make sequential
        self.conv_block = nn.Sequential(*block)


    def forward(self, x):

        output = self.conv_block(x)

        return output

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, name=None):
        super(ConvBlock3D, self).__init__()

        block3D = []
        # first conv layer
        block3D.append(nn.Conv3d(in_channels, out_channels, kernel_size,
                                 padding=padding, stride=stride))
        block3D.append(nn.ReLU())
        block3D.append(nn.BatchNorm3d(out_channels))

        # second conv layer
        block3D.append(nn.Conv3d(out_channels, out_channels, kernel_size,
                                 padding=padding, stride=stride))
        block3D.append(nn.ReLU())
        block3D.append(nn.BatchNorm3d(out_channels))

        # make sequential
        self.conv_block3D = nn.Sequential(*block3D)

    def forward(self, x):
        output = self.conv_block3D(x)
        return output


class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(DownSampling, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        output = self.max_pool(conv_out)

        return output, conv_out

class DownSampling3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(DownSampling3D, self).__init__()

        self.conv = ConvBlock3D(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        output = self.max_pool(conv_out)

        return output, conv_out


class UpSampling(nn.Module):
    '''
    Concat inside.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(UpSampling, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.conv_t = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, \
                                         padding=1, stride=2, output_padding=1)


    def forward(self, x, skip):

        conv_out = self.conv(x)
        output = self.conv_t(conv_out)

        output += skip

        return output

class UpSampling3D(nn.Module):
    '''Concat inside.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(UpSampling3D, self).__init__()

        self.conv = ConvBlock3D(in_channels, out_channels, kernel_size)
        self.conv_t = nn.ConvTranspose3d(out_channels, out_channels, kernel_size,
                                         padding=1, stride=2, output_padding=1)

    def forward(self, x, skip):
        conv_out = self.conv(x)
        output = self.conv_t(conv_out)
        output += skip

        return output

