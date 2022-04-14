import torch.nn as nn
from nets.convNd import convNd
import torch
import torch.nn.functional as F


class Padder(nn.Module):
    def __init__(self, pad_len, mode='constant', value=0.5):
        assert mode == 'constant'  # 目前仅支持 constant
        super(Padder, self).__init__()
        self.pad_len = pad_len
        self.mode = mode
        self.value = value

    def forward(self, pic_in):
        pad_len = int(self.pad_len / 2)
        pad = (pad_len, pad_len, pad_len, pad_len)
        pic_out = F.pad(pic_in, pad, self.mode, self.value)
        return pic_out


class ConvKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='ReLu'):
        assert activation == 'ReLu' or activation == 'Sigmoid' or activation == 'None'
        super(ConvKernel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel = convNd(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             num_dims=4,
                             kernel_size=self.kernel_size,
                             stride=1,
                             padding=0,
                             use_bias=False)
        if self.activation == 'Sigmoid':
            self.Activation = nn.Sigmoid()
        if self.activation == 'ReLu':
            self.Activation = nn.ReLU(inplace=True)
        if self.activation == 'None':
            self.Activation = lambda x: x

    def forward(self, pic_in):
        pic_out = self.Activation(self.kernel(pic_in))
        return pic_out


class PixelShuffle(nn.Module):
    def __init__(self, factor, type):
        assert type == 'ang' or type == 'spa'
        super(PixelShuffle, self).__init__()
        self.factor = factor
        self.type = type

    def forward(self, pic_in):
        size = pic_in.size()
        if self.type == 'spa':
            pic_out = torch.zeros((size[0],
                                   size[1] // (self.factor ** 2),
                                   size[2],
                                   size[3],
                                   size[4] * self.factor,
                                   size[5] * self.factor)).to(pic_in.device)
            for idx_ang_x in range(size[2]):
                for idx_ang_y in range(size[3]):
                    buffer = pic_in[:, :, idx_ang_x, idx_ang_y, :, :]
                    shuffled = torch.pixel_shuffle(buffer, self.factor)
                    pic_out[:, :, idx_ang_x, idx_ang_y, :, :] = shuffled[:, :, :, :]
            return pic_out
        if self.type == 'ang':
            assert size[2] == 1 and size[3] == 1
            pic_out = torch.zeros((size[0],
                                   size[1] // (self.factor ** 2),
                                   size[2] * self.factor,
                                   size[3] * self.factor,
                                   size[4],
                                   size[5])).to(pic_in.device)
            idx_channel = 0
            for idx_ang_x in range(self.factor):
                for idx_ang_y in range(self.factor):
                    pic_out[:, :, idx_ang_x, idx_ang_y, :, :] = pic_in[:, idx_channel:idx_channel + 1, 0, 0, :, :]
                    idx_channel += 1
            return pic_out
