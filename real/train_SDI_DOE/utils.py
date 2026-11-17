import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
from architecture import *
from fvcore.nn import FlopCountAnalysis

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand'] / 65536.
        elif "HSI" in img_dict:
            img = img_dict['HSI']
        else:
            raise AttributeError('Uknown data temp')
        img = np.array(img).astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 586, 586, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        if 'HSI_crop_926' in img_dict:
            img = img_dict['HSI_crop_926'] / np.max(img_dict['HSI_crop_926'])
        if 'HSI_crop_586' in img_dict:
            img = img_dict['HSI_crop_586'] / np.max(img_dict['HSI_crop_586'])
        elif 'HSI_crop_256' in img_dict:
            img = img_dict['HSI_crop_256'] / np.max(img_dict['HSI_crop_256'])
        test_data[i, :, :, :] = img
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def init_filter_function(size):
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

def init_psf(psf_path):
    psf = np.load(psf_path).astype(np.float32)
    [C, h, w] = psf.shape
    for i in range(C):
        psf[i] = psf[i] / (np.sum(psf[i]))
    return psf

def init_psf_HSFAUT(psf_path):
    psf = np.load(psf_path).astype(np.float32)
    [C, h, w] = psf.shape
    psf = psf[:, 256:768, 256:768]
    for i in range(C):
        psf[i] = psf[i] / (np.sum(psf[i]))
    return psf

def init_DOE_psf(psf_path):
    dict = sio.loadmat(psf_path)
    psf = dict['gray']
    psf = np.asarray(psf).astype(np.float32)
    psf = psf.transpose(2, 0, 1)
    psf = psf / psf.max()
    for i in range(psf.shape[0]):
        psf[i] = psf[i] / (np.sum(psf[i]))
    return psf


def torch_psnr(img, ref):
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def init_input_adis(test_gt, filter_function, psf, crop_size):
    [bs, C, YH, YW] = test_gt.shape
    [C, h, w] = filter_function.shape

    offset1 = (YH - crop_size) // 2
    input_cube = test_gt[:, :, offset1:offset1+crop_size, offset1:offset1+crop_size]

    spa_offset = (crop_size - h) // 2
    gt_batch = input_cube[:, :, spa_offset:spa_offset+h, spa_offset:spa_offset+w]

    ff_batch = np.repeat(filter_function[np.newaxis, :, :, :], bs, axis=0)
    ff_batch = torch.FloatTensor(ff_batch)

    psf_batch = np.repeat(psf[np.newaxis, :, :, :], bs, axis=0)
    psf_batch = torch.FloatTensor(psf_batch)

    input_cube = input_cube.cuda().float()
    gt_batch = gt_batch.cuda().float()
    ff_batch = ff_batch.cuda().float()
    psf_batch = psf_batch.cuda().float()
    return input_cube, gt_batch, ff_batch, psf_batch

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def Net_para_calculate(test_model, H = 256, W = 256, C = 28, N = 1):
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, 1, 256, 256)).cuda()
    y_l = torch.randn((1,28,256,256)).cuda()
    pattern = torch.randn((N, C, 512, 512)).cuda()


    flops = FlopCountAnalysis(model,(inputs, y_l, pattern))
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

if __name__ == '__main__':

    model = model_generator('HSFAUT_3stg')
    Net_para_calculate(model)
