import torch
import torch.nn as nn
from nets.utils import Padder, PixelShuffle, ConvKernel


class DimNet_Plus(nn.Module):
    def __init__(self, n_block, ang_res, receptive_field, upscale_factor, channels, activation):
        super(DimNet_Plus, self).__init__()
        self.n_block = n_block
        self.ang_res = ang_res
        self.receptive_field = receptive_field
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.activation = activation

        self.dim_group = DimGroup(self.n_block, self.ang_res, self.receptive_field, self.channels, self.activation)
        self.bottle_neck = BottleNeck(self.ang_res, self.channels, self.upscale_factor, self.activation)

    def model_filename(self):
        return 'DimNet_Plus_b' + str(self.n_block) + '_a' + str(self.ang_res) + '_r' + str(
            self.receptive_field) + '_u' + str(self.upscale_factor) + '_c' + str(
            self.channels) + '_' + self.activation + '.pth.tar'

    def forward(self, pic_in):
        buffer = pic_in.unsqueeze(1)
        buffer = buffer.expand((-1, self.channels, -1, -1, -1, -1))
        buffer = self.dim_group(buffer)
        pic_out = self.bottle_neck(buffer)
        pic_out = pic_out.squeeze(1)
        return pic_out


class DimBlock(nn.Module):
    def __init__(self, in_channels, out_channels, receptive_field, ang_res, activation):
        super(DimBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = receptive_field
        self.ang_res = ang_res
        self.activation = activation
        self.padder = Padder(receptive_field - 1)
        self.kernel = ConvKernel(in_channels=self.in_channels,
                                 out_channels=self.out_channels * self.ang_res ** 2,
                                 kernel_size=(self.ang_res, self.ang_res, self.receptive_field, self.receptive_field),
                                 activation=self.activation)
        self.pixel_shuffle = PixelShuffle(factor=self.ang_res,
                                          type='ang')

    def forward(self, pic_in):
        buffer = self.padder(pic_in)
        buffer = self.kernel(buffer)
        pic_out = self.pixel_shuffle(buffer)
        return pic_out


class DimGroup(nn.Module):
    def __init__(self, n_block, ang_res, receptive_field, channels, activation):
        super(DimGroup, self).__init__()
        self.n_block = n_block
        self.receptive_field = receptive_field
        self.channels = channels
        self.ang_res = ang_res
        self.activation = activation
        modules = []
        for i in range(self.n_block):
            modules.append(DimBlock(in_channels=self.channels,
                                    out_channels=self.channels,
                                    ang_res=self.ang_res,
                                    receptive_field=self.receptive_field,
                                    activation=self.activation))
        self.chanined_blocks = nn.Sequential(*modules)

    def forward(self, pic_in):
        buffer = pic_in
        for i in range(self.n_block):
            buffer = self.chanined_blocks[i](buffer)
        pic_out = buffer
        return pic_out


class BottleNeck(nn.Module):
    def __init__(self, ang_res, channels, upscale_factor, activation):
        super(BottleNeck, self).__init__()
        self.ang_res = ang_res
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.activation = activation
        self.dim_block = DimBlock(self.channels, self.upscale_factor ** 2, 1, self.ang_res, self.activation)
        self.pixel_shuffle = PixelShuffle(self.upscale_factor, 'spa')

    def forward(self, pic_in):
        buffer = self.dim_block(pic_in)
        pic_out = self.pixel_shuffle(buffer)
        return pic_out


if __name__ == '__main__':
    tmp = torch.zeros((1, 5, 5, 80, 80)).to('cuda')
    net = DimNet_Plus(n_block=3,
                      ang_res=5,
                      receptive_field=9,
                      upscale_factor=4,
                      channels=8,
                      activation='ReLu').to('cuda')
    out = net(tmp)
    print(out.size())
