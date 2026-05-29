import torch
from .TSA_Net import TSA_Net
from .MST_Plus_Plus import MST_Plus_Plus
from .Lambda_Net import Lambda_Net
from .CSST import CSST
from .Restormer import Restormer
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .U_Net import U_Net
from .HSFAUT import HSFAUT
from .fftformer import fftformer
from .NAFNet import NAFNet





def model_generator(method, pretrained_model_path=None):
    if method == 'tsa_net':
        model = TSA_Net(in_ch=28, out_ch=28).cuda()
    elif method == 'mirnet':
        model = MIRNet(in_channels=28, out_channels=28, n_feat=28).cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'lambda_net':
        model = Lambda_Net(out_ch=28).cuda()
    elif 'CSST' in method:
        num_iterations = int(method.split('_')[1][0])  #5
        model = CSST(num_iterations=num_iterations).cuda()
    elif 'HSFAUT' in method:
        if 'Ablation' in method:
            print('Ablation study on HSFAUT')
        else:
            if 'HSFAUT' in method:
                num_iterations = int(method.split('_')[1][0])  # 5
                model = HSFAUT(in_channels=28, out_channels=28, n_feat=28, num_iterations=num_iterations).cuda()
            else:
                num_iterations = int(method.split('_')[1][0])  # 5
                model = HSFAUT(in_channels=28, out_channels=28, n_feat=28, num_iterations=num_iterations).cuda()

    elif 'restormer' in method:
        model = Restormer(inp_channels=28, out_channels=28).cuda()
    elif 'fftformer' in method:
        model = fftformer(inp_channels=28, out_channels=28).cuda()
    elif method == 'NAFNet':
        model = NAFNet(img_channel=28, width=64, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]).cuda()  # 64和原文设置保持一致
    elif method == 'NAFNet_light':
        model = NAFNet(img_channel=28, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).cuda()  # 32和原文设置保持一致

    elif 'mprnet' in method:
        model = MPRNet(in_c=28, out_c=28,  n_feat=28).cuda()
    elif 'unet' in method:
        model = U_Net(in_ch=28, out_ch=28).cuda()

    else:
        raise AttributeError(f'Method {method} is not defined !!!!')
    model = torch.nn.DataParallel(model)
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k: v for k, v in checkpoint.items()},
                              strict=True)
    # if method == 'hdnet':
    #     return model, fdl_loss
    return model
