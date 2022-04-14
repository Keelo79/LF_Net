import torch
import torch.nn as nn
from nets.utils import Padder, ConvKernel, PixelShuffle


class DimNet_Minus(nn.Module):
    def __init__(self, ang_res, receptive_field, upscale_factor, activation):
        assert ang_res % 2 == 1 and (ang_res + 1) // 2 % 2 == 1
        assert receptive_field % 2 == 1 and (receptive_field + 1) // 2 % 2 == 1
        super(DimNet_Minus, self).__init__()
        self.ang_res = ang_res
        self.receptive_field = receptive_field
        self.upscale_factor = upscale_factor
        self.padder = Padder(self.receptive_field - 1)
        self.activation = activation
        kernel_size_1 = (self.ang_res, self.ang_res, self.receptive_field, self.receptive_field)
        kernel_size_2 = ((self.ang_res + 1) // 2, (self.ang_res + 1) // 2, (self.receptive_field + 1) // 2,
                         (self.receptive_field + 1) // 2)
        self.conv_kernel_1 = ConvKernel(1,
                                        self.upscale_factor ** 2 * self.ang_res ** 2,
                                        kernel_size_1,
                                        self.activation)
        self.conv_kernel_2_1 = ConvKernel(1,
                                          self.ang_res * self.upscale_factor,
                                          kernel_size_2,
                                          self.activation)
        self.conv_kernel_2_2 = ConvKernel(self.ang_res * self.upscale_factor,
                                          self.ang_res ** 2 * self.upscale_factor ** 2,
                                          kernel_size_2,
                                          self.activation)
        self.recon_block = ReconBlock(self.ang_res, self.upscale_factor)

    def forward(self, pic_in):
        buffer = pic_in.unsqueeze(1)
        buffer = self.padder(buffer)
        buffer_1 = self.conv_kernel_1(buffer)
        buffer_2 = self.conv_kernel_2_2(self.conv_kernel_2_1(buffer))
        pic_out = self.recon_block(buffer_1, buffer_2)
        pic_out = pic_out.squeeze(1)
        return pic_out

    def model_filename(self):
        return 'DimNet_Minus_a' + str(self.ang_res) + '_r' + str(self.receptive_field) + '_u' + str(
            self.upscale_factor) + '_' + self.activation + '.pth.tar'


class ReconBlock(nn.Module):
    def __init__(self, ang_res, upscale_factor):
        super(ReconBlock, self).__init__()
        self.ang_res = ang_res
        self.upscale_factor = upscale_factor
        self.pixel_shuffle_spa = PixelShuffle(self.upscale_factor, 'spa')
        self.pixel_shuffle_ang = PixelShuffle(self.ang_res, 'ang')

    def forward(self, pic_in_1, pic_in_2):
        buffer = pic_in_1 + pic_in_2
        buffer = self.pixel_shuffle_spa(buffer)
        pic_out = self.pixel_shuffle_ang(buffer)
        return pic_out


if __name__ == '__main__':
    tmp = torch.zeros((1, 5, 5, 80, 80)).to('cuda')
    net = DimNet_Minus(5, 5, 4, 'None').to('cuda')
    with torch.no_grad():
        out = net(tmp)
    print(out.size())
    print(out.device)
