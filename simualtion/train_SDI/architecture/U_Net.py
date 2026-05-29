import math
import torch
from torch import nn
from .model_base import Fast_rFFT2d_GPU_batch, Measurement_Render

class U_Net(nn.Module):
    def __init__(self, in_ch=28, out_ch=28):
        super(U_Net, self).__init__()
        # Encoding
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        self.E1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1), nn.PReLU())  # 1/2 256
        self.E2 = nn.Sequential(nn.Conv2d(in_ch, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())  # 1/4 128
        self.E3 = nn.Sequential(nn.Conv2d(62, 124, kernel_size=4, stride=2, padding=1), nn.PReLU())  # 1/8 64
        self.E4 = nn.Sequential(nn.Conv2d(124, 248, kernel_size=4, stride=2, padding=1), nn.PReLU())  # 1/16 32
        self.E5 = nn.Sequential(nn.Conv2d(248, 496, kernel_size=4, stride=2, padding=1), nn.PReLU())  # 1/32 16
        self.E6 = nn.Sequential(nn.Conv2d(496, 992, kernel_size=3, stride=1, padding=1), nn.PReLU())  # 1/32 16

        # Decoding
        self.D1 = nn.Sequential(nn.Conv2d(992, 496, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.D2 = nn.Sequential(nn.ConvTranspose2d(496 + 496, 496, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D3 = nn.Sequential(nn.ConvTranspose2d(496 + 248, 248, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D4 = nn.Sequential(nn.ConvTranspose2d(248 + 124, 124, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D5 = nn.Sequential(nn.ConvTranspose2d(124 + 62, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D6 = nn.Sequential(nn.ConvTranspose2d(62 + in_ch, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())
        # Projection
        self.project = nn.Conv2d(62, out_ch, kernel_size=1)
        # render
        self.Render = Measurement_Render()
        # add
        self.ff_down = nn.Conv2d(28, 3, 1, stride=1, padding=0)
        # 512
        self.psf_down_512 = nn.Conv2d(28, 3, 2, stride=2, padding=0)
        # 1024
        self.psf_down_1024 = nn.Conv2d(28, 3, 4, stride=4, padding=0)
        self.fution = nn.Conv2d(7, 28, 1, padding=0, bias=True)

    def initial(self, y, ff_batch, psf_batch):
        ff_feature = self.ff_down(ff_batch)  # 3
        # psf_feature = self.psf_down(psf_batch)  # 3
        if psf_batch.shape[3] == 512:
            psf_feature = self.psf_down_512(psf_batch)
        elif psf_batch.shape[3] == 1024:
            psf_feature = self.psf_down_1024(psf_batch)
        else:
            raise AttributeError('ukonwn psf shape')
        z = self.fution(torch.cat([y, ff_feature, psf_feature], dim=1))  # [b,7,256,256] --> [b,28,256,256]
        return z

    def forward(self, input_cube, ff_batch, psf_batch, imaging_system, y=None):
        """
        :param input_cube: [bs, 28, 512, 512]
        :param ff_batch:   [bs, 28, 256, 256]
        :param psf_batch:  [bs, 28, 512, 512]
        :param imaging_system: 'ADIS'
        :return: y:        [bs, 1, 256, 256]
        """
        if y == None:
            y = self.Render(input_cube=input_cube, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system)
        z = self.initial(y, ff_batch, psf_batch)
        # Encoding
        E1 = self.E1(z)   # output size : 128
        E2 = self.E2(E1)  # output size : 64
        E3 = self.E3(E2)  # output size : 32
        E4 = self.E4(E3)  # output size : 16
        E5 = self.E5(E4)  # output size : 8
        E6 = self.E6(E5)  # output size : 8
        # Decoding
        D1 = self.D1(E6)  # input size : 46x46
        D2 = self.D2(torch.cat((D1, E5), dim=1))  # input size : 46x46
        D3 = self.D3(torch.cat((D2, E4), dim=1))  # input size : 92x92
        D4 = self.D4(torch.cat((D3, E3), dim=1))  # input size : 184x184
        D5 = self.D5(torch.cat((D4, E2), dim=1))  # input size : 184x184
        D6 = self.D6(torch.cat((D5, E1), dim=1))  # input size : 184x184
        output = self.project(D6)

        return output

def ReconModel():
    return U_Net()
