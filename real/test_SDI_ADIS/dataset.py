import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio

class dataset(tud.Dataset):
    def __init__(self, opt, train_data1, train_data2, filter_function, psf):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        self.crop_szie = opt.crop_size
        self.batch_size = opt.batch_size
        self.train_data1 = train_data1
        self.train_data2 = train_data2
        self.filter_function = filter_function
        self.psf = psf
        if self.isTrain == True:
            self.num = opt.trainset_num
            self.arguement = True
        else:
            self.num = opt.testset_num
            self.arguement = False
    def Shuffle_Crop(self, train_data1, train_data2):
        train_data_index = np.random.choice(range(2), 1)
        if train_data_index == 0:
            gt_batch = self.Shuffle_Core(train_data1)
        elif train_data_index == 1 :
            gt_batch = self.Shuffle_Core(train_data2)
        else:
            raise AttributeError('Unkown index of datasets, set 0 or 1')
        return gt_batch


    def Arguement(slef, x):
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        for j in range(rotTimes):
            x = np.rot90(x, axes=(1, 2))
        for j in range(vFlip):
            x = np.flip(x, axis=(2,))
        for j in range(hFlip):
            x = np.flip(x, axis=(1,))
        return x

    def Shuffle_Core(self, train_data):
        crop_size = self.crop_szie
        gt_batch = np.zeros((28, crop_size, crop_size), dtype=np.float32)
        processed_data = np.zeros((crop_size, crop_size, 28), dtype=np.float32)
        index = np.random.choice(range(len(train_data)), 1)
        img = train_data[index[0]]
        h, w, _ = img.shape
        x_index = random.randint(0, (h - crop_size))
        y_index = random.randint(0, (w - crop_size))
        processed_data[:, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = processed_data.transpose(2, 0, 1)
        gt_batch[:, :, :] = self.Arguement(processed_data)
        return gt_batch
    def init_input_adis(self, gt_batch):
        filter_function = self.filter_function
        psf = self.psf
        input_cube = gt_batch

        gamma = np.random.choice(np.arange(1.5, 3.1, 0.1))
        input_cube = (input_cube / np.max(input_cube)).astype(np.float32)
        input_cube = np.clip(input_cube, 1e-6, None)
        input_cube = input_cube ** gamma


        spa_offset = (self.crop_szie-self.size) // 2
        GT_cube = input_cube[:, spa_offset:spa_offset+self.size, spa_offset:spa_offset+self.size]
        return input_cube, GT_cube, filter_function, psf

    def __getitem__(self, index):
        if self.isTrain == False:
            print('error, train code here')
        if self.isTrain == True:
            train_data1 = self.train_data1
            train_data2 = self.train_data2
            gt_batch = self.Shuffle_Crop(train_data1, train_data2)
            input_cube, gt_cube, filter_function, psf = self.init_input_adis(gt_batch)


            input_cube = torch.FloatTensor(input_cube.copy())
            gt_cube = torch.FloatTensor(gt_cube.copy())
            filter_function = torch.FloatTensor(filter_function.copy())
            psf = torch.FloatTensor(psf.copy())
        return input_cube, gt_cube, filter_function, psf

    def __len__(self):
        return self.num

