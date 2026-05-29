import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio


class dataset(tud.Dataset):
    def __init__(self, opt, train_data1, train_data2):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size                 # 256
        self.crop_szie = opt.crop_size       # 256
        self.batch_size = opt.batch_size
        self.train_data1 = train_data1
        self.train_data2 = train_data2
        self.imaging_system = opt.imaging_system
        # self.filter_function = filter_function
        # self.psf = psf
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


    def Arguement(slef, x):  # processed_data[i]：（28,512,512）
        """
        :param x: c,h,w
        :return: c,h,w
        """
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        # Random rotation
        for j in range(rotTimes):
            x = np.rot90(x, axes=(1, 2))  
        # Random vertical Flip
        for j in range(vFlip):
            x = np.flip(x, axis=(2,))  
        # Random horizontal Flip
        for j in range(hFlip):
            x = np.flip(x, axis=(1,))  
        return x

    def Arguement2(slef, x):  # processed_data[i]：（4,28,256,256）
        """
        :param x: 4,c,h,w
        :return: c,2h,2w
        """
        c, h, w = x[0].shape
        H, W = 2*h, 2*w
        output = np.zeros((c,H,W), dtype=np.float32)
        output[:, :h, :w] = x[0]
        output[:, :h, w:] = x[1]
        output[:, h:, :w] = x[2]
        output[:, h:, w:] = x[3]
        return output


    def Shuffle_Core(self, train_data):
        crop_size = self.crop_szie
        gt_batch = np.zeros((28, crop_size, crop_size), dtype=np.float32)
        shuffle_type = np.random.choice([0,1], p=[0.5,0.5])
        if shuffle_type == 0:
            processed_data = np.zeros((crop_size, crop_size, 28), dtype=np.float32)
            index = np.random.choice(range(len(train_data)), 1)
            img = train_data[index[0]]
            h, w, _ = img.shape
            x_index = random.randint(0, (h - crop_size))
            y_index = random.randint(0, (w - crop_size))
            processed_data[:, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            processed_data = processed_data.transpose(2, 0, 1)
            gt_batch[:, :, :] = self.Arguement(processed_data)
        else:
            processed_data = np.zeros((4, crop_size//2, crop_size//2, 28), dtype=np.float32)
            sample_list = np.random.randint(0, len(train_data), 4)
            for i in range(4):
                img = train_data[sample_list[i]]
                h, w, _ = img.shape
                x_index = random.randint(0, (h - crop_size//2))
                y_index = random.randint(0, (w - crop_size//2))
                processed_data[i] = img[x_index:x_index + crop_size//2, y_index:y_index + crop_size//2, :]
            gt_batch = self.Arguement2(processed_data.transpose(0, 3, 1, 2))
        return gt_batch
    
    def init_input_adis(self, gt_batch):
        input_cube = gt_batch
        if self.imaging_system == 'diffuser-based':
            mask_cube = np.zeros_like(input_cube, dtype=input_cube.dtype)
            mask_cube[:, 128:384, 128:384] = 1.0
            input_cube = input_cube * mask_cube
        spa_offset = (self.crop_szie-self.size) // 2
        GT_cube = input_cube[:, spa_offset:spa_offset+self.size, spa_offset:spa_offset+self.size]
        return input_cube, GT_cube


    def __getitem__(self, index):
        if self.isTrain == False:
            print('error, train code here')
        if self.isTrain == True:
            train_data1 = self.train_data1
            train_data2 = self.train_data2
            gt_batch = self.Shuffle_Crop(train_data1, train_data2)      # (28,512, 512)
            input_cube, gt_cube = self.init_input_adis(gt_batch)

            input_cube = torch.FloatTensor(input_cube.copy())
            gt_cube = torch.FloatTensor(gt_cube.copy())
        return input_cube, gt_cube


    def __len__(self):
        return self.num   #1250 or 5000

