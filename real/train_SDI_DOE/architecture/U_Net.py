import math
import torch
from torch import nn
from .model_base import Fast_rFFT2d_GPU_batch, Measurement_Render

class U_Net(nn.Module):
    def __init__(self, in_ch=28, out_ch=28):
        super(U_Net, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.E2 = nn.Sequential(nn.Conv2d(in_ch, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.E3 = nn.Sequential(nn.Conv2d(62, 124, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.E4 = nn.Sequential(nn.Conv2d(124, 248, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.E5 = nn.Sequential(nn.Conv2d(248, 496, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.E6 = nn.Sequential(nn.Conv2d(496, 992, kernel_size=3, stride=1, padding=1), nn.PReLU())

        self.D1 = nn.Sequential(nn.Conv2d(992, 496, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.D2 = nn.Sequential(nn.ConvTranspose2d(496 + 496, 496, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D3 = nn.Sequential(nn.ConvTranspose2d(496 + 248, 248, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D4 = nn.Sequential(nn.ConvTranspose2d(248 + 124, 124, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D5 = nn.Sequential(nn.ConvTranspose2d(124 + 62, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.D6 = nn.Sequential(nn.ConvTranspose2d(62 + in_ch, 62, kernel_size=4, stride=2, padding=1), nn.PReLU())
        self.project = nn.Conv2d(62, out_ch, kernel_size=1)
        self.Render = Measurement_Render()
        self.ff_down = nn.Conv2d(28, 3, 1, stride=1, padding=0)
        self.psf_down_512 = nn.Conv2d(28, 3, 2, stride=2, padding=0)
        self.psf_down_1024 = nn.Conv2d(28, 3, 4, stride=4, padding=0)
        self.fution = nn.Conv2d(7, 28, 1, padding=0, bias=True)

    def initial(self, y, ff_batch, psf_batch):
        ff_feature = self.ff_down(ff_batch)
        if psf_batch.shape[3] == 512:
            psf_feature = self.psf_down_512(psf_batch)
        elif psf_batch.shape[3] == 1024:
            psf_feature = self.psf_down_1024(psf_batch)
        else:
            raise AttributeError('ukonwn psf shape')
        z = self.fution(torch.cat([y, ff_feature, psf_feature], dim=1))
        return z

    def forward(self, input_cube, ff_batch, psf_batch, imaging_system, y=None):
        if y == None:
            y = self.Render(input_cube=input_cube, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system)
        z = self.initial(y, ff_batch, psf_batch)
        E1 = self.E1(z)
        E2 = self.E2(E1)
        E3 = self.E3(E2)
        E4 = self.E4(E3)
        E5 = self.E5(E4)
        E6 = self.E6(E5)
        D1 = self.D1(E6)
        D2 = self.D2(torch.cat((D1, E5), dim=1))
        D3 = self.D3(torch.cat((D2, E4), dim=1))
        D4 = self.D4(torch.cat((D3, E3), dim=1))
        D5 = self.D5(torch.cat((D4, E2), dim=1))
        D6 = self.D6(torch.cat((D5, E1), dim=1))
        output = self.project(D6)

        return output

def ReconModel():
    return U_Net()
