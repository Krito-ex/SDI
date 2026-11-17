import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


from dataset import dataset
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





seed = opt.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

filter_function = init_filter_function(opt.ff_path)
psf = init_psf(opt.psf_path)
print('prior information is loaded')
test_data = LoadTest(opt.test_path)
test_gt = test_data.cuda().float()
print('dataset is loaded')
imaging_system = opt.imaging_system

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf
if not os.path.exists(result_path):
    os.makedirs(result_path)


if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
    model = model.cuda()
    FDL_loss = FDL_loss.cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()


def test(logger, input_mea, gt_batch, ff_batch, psf_batch):
    psnr_list, ssim_list = [], []

    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system, y=input_mea)
        elif 'CSST' in opt.method:
            model_out = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system, y=input_mea)
        else:
            model_out = model(input_cube=None, ff_batch=ff_batch, psf_batch=psf_batch, imaging_system=imaging_system, y=input_mea)

    end = time.time()
    for k in range(gt_batch.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], gt_batch[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], gt_batch[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(gt_batch.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(result_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    input_mea, gt_batch, ff_batch, psf_batch = init_input_adis(test_gt, filter_function, psf, opt.crop_size, opt.imaging_system)

    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(logger, input_mea, gt_batch, ff_batch, psf_batch)
    if psnr_mean > psnr_max:
        psnr_max = psnr_mean
        print(np.asarray(psnr_all))
        if psnr_mean > 20:
            name = result_path + '/' + 'Test_{:.2f}_{:.3f}'.format(psnr_max, ssim_mean) + '.mat'
            scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()




