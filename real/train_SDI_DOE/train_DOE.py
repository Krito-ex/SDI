import os
from option import opt
from dataset import dataset
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
import torch.utils.data as tud



os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

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

train_set1 = LoadTraining(opt.data_path1)
train_set2 = LoadTraining(opt.data_path2)
print('dataset is loaded')
filter_function = init_filter_function(opt.size)

psf = init_DOE_psf(opt.psf_path)

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

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

def train(epoch, logger, psf, ff):
    model.train()
    Dataset = dataset(opt, train_set1, train_set2)
    loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    epoch_loss = 0
    start_time = time.time()

    psf = torch.FloatTensor(psf.copy())
    psf = Variable(psf)
    ff = torch.FloatTensor(ff.copy())
    ff = Variable(ff)

    for i, (input_cube, gt_batch) in enumerate(loader_train):
        input_cube, gt_batch = Variable(input_cube), Variable(gt_batch)
        input_cube, gt_batch = input_cube.cuda(), gt_batch.cuda()
        psf_batch = psf.unsqueeze(0).repeat(input_cube.shape[0], 1, 1, 1)
        ff_batch = ff.unsqueeze(0).repeat(input_cube.shape[0], 1, 1, 1)
        psf_batch = psf_batch.to(input_cube.device)
        ff_batch = ff_batch.to(input_cube.device)

        optimizer.zero_grad(set_to_none=True)
        if 'CSST' in opt.method:
            model_out = model(input_cube, ff_batch, psf_batch)
            loss = torch.sqrt(mse(model_out, gt_batch))
        elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, diff_pred = model(input_cube, ff_batch, psf_batch)
            loss = torch.sqrt(mse(model_out, gt_batch))
            diff_gt = torch.mean(torch.abs(model_out.detach() - gt_batch), dim=1, keepdim=True)
            loss_sparsity = F.mse_loss(diff_gt, diff_pred)
            loss = loss + 2 * loss_sparsity
        else:
            model_out = model(input_cube, ff_batch, psf_batch)
            loss = torch.sqrt(mse(model_out, gt_batch))
        if opt.method == 'hdnet':
            fdl_loss = FDL_loss(model_out, gt_batch)
            loss = loss + 0.7 * fdl_loss
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    scheduler.step()
    end_time = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / len(Dataset), (end_time - start_time)))
    return 0

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger, psf, filter_function)
        if epoch % 10 == 0:
            checkpoint(model, epoch, model_path, logger)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()




