import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from .model_base import Fast_rFFT2d_GPU_batch, Measurement_Render

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out

class SFA_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            use_freq=True,
            reduction_ratio=4
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.use_freq = use_freq

        if use_freq:
            self.freq_conv = nn.Sequential(
                nn.Conv2d(dim, dim // reduction_ratio, 1),  # 压缩通道
                nn.GELU(),
                nn.Conv2d(dim // reduction_ratio, dim, 1)   # 恢复通道
            )
            self.freq_gate = nn.Parameter(torch.tensor([0.1]))  # 自适应融合权重

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        if self.use_freq:
            x_perm = x_in.permute(0, 3, 1, 2)
            freqs = torch.fft.rfft2(x_perm, norm='ortho')
            amp = torch.abs(freqs)
            freq_feat = self.freq_conv(amp)
            if freq_feat.size(-1) != w:  # rfft输出尺寸变化处理
                freq_feat = F.interpolate(freq_feat, size=(h, w), mode='bilinear')
            freq_feat = freq_feat.permute(0, 2, 3, 1)
            out_freq = self.freq_gate * freq_feat

        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        if self.use_freq:
            out += out_freq
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SFAAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SFA_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SFAT(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, stage=2, num_blocks=[2,4,4]):
        super(SFAT, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                SFAAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SFAAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                SFAAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding    需要和SST的方法保持一致
        fea = self.embedding(x)
        x = x[:,:self.out_dim,:,:]

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out


class HyPaNet(nn.Module):
    def __init__(self, in_nc, para_nums, num_iterations, channel=256):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, para_nums*num_iterations, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = para_nums*num_iterations

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x


def Fourier_Tansform(cube):
    cube_fft = torch.fft.fft2(cube)
    cube_fft = torch.fft.fftshift(cube_fft)
    return cube_fft

def Inverse_Fourier_Transform(cube_fft):
    cube_fft = torch.fft.ifftshift(cube_fft)
    cube = torch.fft.ifft2(cube_fft)
    return cube


def phi_filter(x, filter_function, imaging_system):
    para = {'ADIS': 15, 'DOE-based': 4, 'diffuser-based': 6}
    filter_function = filter_function * para[imaging_system]
    x = x * filter_function
    y = torch.sum(x, dim=1, keepdim=True)
    return y

def phi_filter_t(y, filter_function, imaging_system):
    para = {'ADIS': 15, 'DOE-based': 4, 'diffuser-based': 6}
    filter_function = filter_function * para[imaging_system]
    x = y.repeat(1, filter_function.shape[1], 1, 1)
    x = x * filter_function
    return x

# def phi_filter_tnt(x, filter_function, imaging_system):
#     para = {'ADIS': 15, 'DOE-based': 4, 'diffuser-based': 6}
#     filter_function = filter_function * para[imaging_system]
#     y = torch.sum(x * filter_function, dim=1, keepdim=True)
#     x = y.repeat(1, filter_function.shape[1], 1, 1) * filter_function
#     return x

def phi_filter_tnt(filter_function, imaging_system):
    para = {'ADIS': 15, 'DOE-based': 4, 'diffuser-based': 6}
    filter_function = filter_function * para[imaging_system]
    phi_filter_s = torch.sum(filter_function * filter_function, dim=1, keepdim=True) / (filter_function.shape[1]*filter_function.shape[1])
    return phi_filter_s



def Fast_conv(DHSI, psf_cube):
    '''
    Args:
        DHSI: [bs, C, H, W]        256
        psf_cube: [bs, C, h, w]    512
    Returns:
        output_cube: [bs, C, H, W]
    not crop
    '''
    [bs,C,H,W] = DHSI.shape
    [bs,C,h,w] = psf_cube.shape
    output_cube = torch.zeros([bs, C, H, W], device=DHSI.device)
    # prepare padding size to accommodate FFT-based convolution
    YH = max(H, h)
    YW = max(W, w)
    for i in range(bs):
        HSI = DHSI[i]   # [C,H,W]
        psf = psf_cube[i]   # [C,h,w]
        HSI_padded = torch.zeros([C, YH, YW], device=HSI.device)
        psf_padded = torch.zeros([C, YH, YW], device=psf.device)
        HSI_padded[:, (YH - H) // 2:(YH - H) // 2 + H, (YW - W) // 2:(YW - W) // 2 + W] = HSI
        psf_padded[:, (YH - h) // 2:(YH - h) // 2 + h, (YW - w) // 2:(YW - w) // 2 + w] = psf
        HSI_fft = torch.fft.rfft2(HSI_padded, s=(YH,YW)).cuda()
        psf_fft = torch.fft.rfft2(psf_padded, s=(YH,YW)).cuda()
        # Element-wise multiplication in the frequency domain
        conv_result_fft = HSI_fft * psf_fft
        # inverse FFT to get the convolution result in the spatial domain
        conv_result = torch.fft.irfft2(conv_result_fft)[:,:YH,:YW]
        # roll
        conv_result = torch.roll(conv_result, shifts=(YH//2, YW//2), dims=(1,2))
        # crop the result to the valid region
        meas = conv_result[:, YH//2-H//2:YH//2+H//2, YW//2-W//2:YW//2+W//2]
        output_cube[i] = meas.squeeze(0)
    return output_cube


def phi_conv(u, psf_batch):
    u_conv = Fast_conv(u, psf_batch)
    return u_conv

def phi_conv_f(x_f, psf_batch_f):
    result = x_f * psf_batch_f
    return result

def phi_conv_t_f(x_f, psf_batch_f):
    result = x_f * psf_batch_f
    return result

def pad(x):
    bs, nC, h, w = x.shape
    H, W = 512, 512
    pad_w = (W-w)//2
    pad_h = (H-h)//2
    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    return x_padded



def crop(x):
    bs, nC, H, W = x.shape
    h, w = 256, 256
    start_h = (H-h)//2
    start_w = (W-w)//2
    x_crop = x[:,:,start_h:start_h+h, start_w:start_w+w]
    return x_crop

class HSFAUT(nn.Module):
    def __init__(self, in_channels=3, out_channels=28, n_feat=28, num_iterations=3):
        super(HSFAUT, self).__init__()
        self.num_iterations = num_iterations
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.fution_update_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.fution_update = nn.ModuleList([])
        for _ in range(self.num_iterations):
            self.fution_update.append(SFAAB(dim=n_feat, num_blocks = 3, dim_head=n_feat, heads=1))

        self.denoisers_conv = nn.ModuleList([])
        for _ in range(self.num_iterations):
            self.denoisers_conv.append(SFAT(in_dim=35, out_dim=out_channels, dim=n_feat, stage=2, num_blocks=[1, 1, 1]))

        self.para_estimator_filter = HyPaNet(in_nc=28, para_nums=1, num_iterations=num_iterations)
        self.para_estimator_conv = HyPaNet(in_nc=3, para_nums=2, num_iterations=1)  # 1个复数参数

        # add
        self.ff_down = nn.Conv2d(28, 3, 1, stride=1, padding=0)
        # 512
        self.psf_down_512 = nn.Conv2d(28, 3, 2, stride=2, padding=0)
        self.psf_down_512_f = nn.Conv2d(56, 56, 2, stride=2, padding=0)
        self.psf_down_256_f = nn.Conv2d(56, 6, 1, stride=1, padding=0)
        self.fution = nn.Conv2d(7, 28, 1, padding=0, bias=True)
        self.para_conv = nn.Conv2d(62, 3, 1, padding=0, bias=True)


    def initial(self, y, ff_batch, psf_batch):
        ff_feature = self.ff_down(ff_batch)  # 3
        psf_feature = self.psf_down_512(psf_batch)
        x = self.fution(torch.cat([y, ff_feature, psf_feature], dim=1))  # [b,7,256,256] --> [b,28,256,256]
        parameters = self.para_estimator_filter(self.fution(torch.cat([y, ff_feature, psf_feature], dim=1)))
        betas = parameters[:, :self.num_iterations,:,:]
        return x, betas


    def forward(self, input_cube, ff_batch, psf_batch, imaging_system, y=None):

        """
        :param input_cube: [bs, 28, 512, 512]
        :param ff_batch:   [bs, 28, 256, 256]
        :param psf_batch:  [bs, 28, 512, 512]
        :return: z:        [bs, 28, 256, 256]
        """
        if y==None:
            Render = Measurement_Render()
            y = Render(input_cube=input_cube, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system)   # y=M
        bs, nC, H, W = ff_batch.shape
        x, betas = self.initial(y, ff_batch, psf_batch)
        x, betas = x.contiguous(),  betas.contiguous()
        u = x
        u_f, psf_batch_f = Fourier_Tansform(u), Fourier_Tansform(psf_batch)   
        psf_batch_f_RI = self.psf_down_512_f(torch.cat([psf_batch_f.real, psf_batch_f.imag],dim=1))   
        psf_batch_f = torch.complex(psf_batch_f_RI[:, :nC,:,:], psf_batch_f_RI[:, nC:, :, :])  
        psf_batch_f_s = psf_batch_f * psf_batch_f.conj()                                              
        psf_feature_f_RI = self.psf_down_256_f(psf_batch_f_RI)                                 
        phi_filter_s = phi_filter_tnt(ff_batch, imaging_system)
        
        # dict = []
        x = self.conv_in(x)
        for i in range(self.num_iterations):
            beta = betas[:, i:i + 1, :, :]
            '''stage 1'''
            x_conv = Fast_conv(x, psf_batch)
            a = torch.div(y - phi_filter(x_conv, ff_batch, imaging_system), beta + phi_filter_s)
            j = x_conv + phi_filter_t(a, ff_batch, imaging_system)
            '''fution'''
            a = self.fution_update_conv(torch.cat([j, x_conv], dim=1))
            j = self.fution_update[i](a)
            '''stage 2'''
            # Transform
            j_f = Fourier_Tansform(j)   # 28 256 256
            # estimator
            parameters = self.para_estimator_conv(self.para_conv(torch.cat([j_f.real, j_f.imag, psf_feature_f_RI], dim=1)))  # 62
            varphi, chi = parameters[:, 0:1, :, :], parameters[:, 1:2, :, :]
            u_f = Fourier_Tansform(u).contiguous()
            x_f = torch.div(j_f * psf_batch_f.conj() + varphi * u_f, varphi + psf_batch_f_s)
            # denoise 1
            x = torch.real(Inverse_Fourier_Transform(x_f))
            chi_repeat = chi.repeat(1, 1, x_f.shape[2], x_f.shape[3])  # 1
            u = self.denoisers_conv[i](torch.cat([x, chi_repeat, psf_feature_f_RI], dim=1))  # 28 1 6
        return u[:, :, :, :]
    
