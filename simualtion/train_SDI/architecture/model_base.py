import torch
import torch.nn as nn

@torch.no_grad()
def Fast_rFFT2d_GPU_batch(DHSI, psf_cube):
    '''
    Args:
        DHSI: [bs, C, H, W]
        psf_cube: [bs, C, h, w]
    Returns:
        output_cube: [bs, C, H, W]
    not crop
    '''
    [bs,C,H,W] = DHSI.shape
    [bs,C,h,w] = psf_cube.shape
    if h<1024 or w<1024:   
        PSF = torch.zeros((bs, C, 1024, 1024), dtype=psf_cube.dtype, device=psf_cube.device)
        PSF[:, :, (1024 - h) // 2:(1024 - h) // 2 + h, (1024 - w) // 2:(1024 - w) // 2 + w] = psf_cube
        psf_cube = PSF
    [bs, C, h, w] = psf_cube.shape
    output_cube = torch.zeros([bs, C, H, W], device=DHSI.device)
    # prepare padding size to accommodate FFT-based convolution
    YH = H + h - 1
    YW = W + w - 1
    YH = 2 ** (int(torch.log2(torch.tensor(YH)))+1)
    YW = 2 ** (int(torch.log2(torch.tensor(YW)))+1)
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


class Measurement_Render(nn.Module):
    def __init__(self):
        super(Measurement_Render, self).__init__()

    
    @torch.no_grad()
    def forward(self, input_cube, ff_batch, psf_batch, imaging_system):
        conv_cube = Fast_rFFT2d_GPU_batch(input_cube, psf_batch)
        size = ff_batch.shape[2]
        crop_size = input_cube.shape[2]
        spa_offset = (crop_size - size) // 2
        filtered_cube = conv_cube[:, :, spa_offset:spa_offset+size, spa_offset:spa_offset+size]
        filtered_cube = filtered_cube * ff_batch
        Measurement = torch.sum(filtered_cube, dim=1, keepdim=True)
        para = {'ADIS': 15, 'DOE-based': 4, 'diffuser-based': 6}
        Measurement = Measurement / float(input_cube.shape[1]) * para[imaging_system]
        return Measurement