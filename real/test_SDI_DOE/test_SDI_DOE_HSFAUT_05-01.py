import os
import sys
import argparse
from architecture import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train_SDI_DOE'))
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
import torch.utils.data as tud
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="PyTorch ADIS simulation Test")
parser.add_argument('--template', default='HSFAUT_3stg', help='You can set various')
parser.add_argument('--method', type=str, default='HSFAUT_3stg', help='method name')
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--size", default=256, type=int, help='the size of trainset image')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_name", required=True, type=str)
parser.add_argument("--psf_name", default='DOE_raw512_slim.mat', type=str, help='psf name')
parser.add_argument('--data_root', default='./real/train_SDI_DOE/exp/', type=str,help='path of data')
opt = parser.parse_args()
print(opt)


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

seed = opt.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def init_filter_function_51S5C(size):
    filter_function = np.ones((28, size, size), dtype=np.float32)
    para_R = np.array([
        0.0272, 0.0255, 0.0241, 0.0229, 0.0216, 0.0203, 0.0195, 0.0240, 0.0357, 0.0501,
        0.0649, 0.0769, 0.0745, 0.0607, 0.0452, 0.0339, 0.0461, 0.0880, 0.3664, 0.6803,
        0.8708, 0.9749, 1.0001, 0.9885, 0.9636, 0.9424, 0.9244, 0.9020], dtype=np.float32)
    para_G = np.array([
        0.0765, 0.0789, 0.1210, 0.2421, 0.4504, 0.6939, 0.8183, 0.8941, 0.9316, 0.9575,
        0.9845, 1.0001, 0.9979, 0.9904, 0.9473, 0.8942, 0.8529, 0.8209, 0.7791, 0.7242,
        0.6644, 0.5958, 0.4474, 0.3438, 0.2739, 0.2264, 0.1943, 0.1632], dtype=np.float32)
    para_B = np.array([
        0.9934, 1.0005, 0.9935, 0.9662, 0.9206, 0.8464, 0.7498, 0.6008, 0.5197, 0.3641,
        0.2991, 0.2281, 0.1779, 0.1396, 0.1148, 0.0876, 0.0662, 0.0518, 0.0419, 0.0378,
        0.0361, 0.0327, 0.0286, 0.0242, 0.0196, 0.0172, 0.0225, 0.0380], dtype=np.float32)
    filter_function[:, 0::2, 0::2] *= para_G.reshape(-1, 1, 1)
    filter_function[:, 0::2, 1::2] *= para_B.reshape(-1, 1, 1)
    filter_function[:, 1::2, 0::2] *= para_R.reshape(-1, 1, 1)
    filter_function[:, 1::2, 1::2] *= para_G.reshape(-1, 1, 1)
    return filter_function



filter_function = init_filter_function_51S5C(opt.size)
opt.psf_path = "./real/test_SDI_DOE/real_PSF/" + opt.psf_name
psf = init_DOE_psf(opt.psf_path)

pretrained_model_path = os.path.join(opt.data_root, opt.pretrained_model_name)
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, pretrained_model_path)
    model = model.cuda()
    FDL_loss = FDL_loss.cuda()
else:
    model = model_generator(opt.method, pretrained_model_path).cuda()


def test(h_idx, w_idx, logger, ff_batch, psf_batch, test_measurement):
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, y=test_measurement)
        elif 'CSST' in opt.method:
            model_out = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, y=test_measurement)
        else:
            model_out = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, y=test_measurement)
    end = time.time()
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    logger.info('===> h_idx {}, w_idx {}: time: {:.2f}'.format(h_idx, w_idx, (end - begin)))
    model.train()
    return pred

def main(ff, psf, img_name, img_root):
    logger = gen_log('./real/test_SDI_DOE/')

    img_path = img_root + img_name

    save_root = './real/test_SDI_DOE/results/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    
    


    img = np.load(img_path).astype(np.float32)




    patch = opt.size
    save_patch = opt.size//2
    pad_size = (patch - save_patch)//2
    h, w = img.shape
    h_num, w_num = h//save_patch, w//save_patch
    hh, ww = h_num * save_patch, w_num*save_patch
    img = img[(h-hh)//2:(h-hh)//2+hh, (w-ww)//2:(w-ww)//2+ww]
    padded_img = np.zeros((hh+2*pad_size, ww+2*pad_size), dtype=img.dtype)
    padded_img[pad_size:pad_size + hh, pad_size:pad_size + ww] = img
    padded_img = torch.FloatTensor(padded_img).cuda()
    padded_img = padded_img.unsqueeze(0).unsqueeze(0)
    result_cube = np.zeros((28, hh, ww), dtype=np.float32)

    psf = torch.FloatTensor(psf.copy())
    psf = Variable(psf)
    ff = torch.FloatTensor(ff.copy())
    ff = Variable(ff)


    for i in range(h_num):
        for j in range(w_num):
            mea = padded_img[:, :, i*save_patch:i*save_patch+patch, j*save_patch:j*save_patch+patch]
            psf_batch = psf.unsqueeze(0).repeat(mea.shape[0], 1, 1, 1)
            ff_batch = ff.unsqueeze(0).repeat(mea.shape[0], 1, 1, 1)
            psf_batch = psf_batch.to(mea.device)
            ff_batch = ff_batch.to(mea.device)
            pred = test(h_idx=i, w_idx=j, logger=logger, ff_batch=ff_batch, psf_batch=psf_batch, test_measurement=mea)
            pred = np.transpose(pred, (0, 3, 1, 2))
            result_cube[:, i*save_patch:(i+1)*save_patch, j*save_patch:(j+1)*save_patch] =\
                pred[0, :, pad_size:pad_size+save_patch, pad_size:pad_size+save_patch]


    result_cube = result_cube[:, 64:-64, 64:-64]
    sio.savemat(os.path.join(save_root, img_name.split('.')[0] + '.mat'), {'pred': result_cube})
    return 0


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    mea_path = './real/test_SDI_DOE/real_capture/'
    filenames = os.listdir(mea_path)
    filenames.sort()
    for filename in filenames:
        if filename.endswith('_croped.npy'):
            main(ff=filter_function, psf=psf, img_name=filename, img_root=mea_path)




