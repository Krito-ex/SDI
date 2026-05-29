
import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='HSCUF', help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='4,5')

# Data specifications
parser.add_argument('--data_root', type=str, default='./datasets/', help='dataset directory')


# Model specifications
parser.add_argument('--method', type=str, default='HSFAUTv2_3stg', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')


# Training specifications
parser.add_argument('--imaging_system', type=str, default='ADIS', choices=('ADIS','DOE-based', 'diffuser-based', 'prism-based'), help='Imaging system')
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--size", default=256, type=int, help='reconstruction size')
parser.add_argument("--crop_size", default=512, type=int, help='cropped patch size')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')

opt = parser.parse_args()   #args
template.set_template(opt)
opt.trainset_num = 20000 // ((opt.size // 128) ** 2)     #5000

opt.outf = './exp/' + opt.method +'/'

# init specfic param
opt.ff_name = 'filter_function_450-650-28.npy'
if opt.imaging_system == 'ADIS':
    opt.psf_name = 'New-200um-512-0.5.npy'
elif opt.imaging_system == 'DOE-based':
    opt.psf_name = 'DOE_psf_450-650-28.npy'
elif opt.imaging_system == 'diffuser-based':
    opt.psf_name = 'diffuser_psf_450-650-28.npy'
    opt.ff_name = 'diffuser_filter_function_450-650-28.npy'
elif opt.imaging_system == 'prism-baed':
    opt.psf_name = 'prism-200um-512-1-450-650-28.npy'
else:
    raise AttributeError('Unknown imaging system type')



opt.data_path1 = f"{opt.data_root}cave_1024_28/"
opt.data_path2 = f"{opt.data_root}KAIST_non_selected/"
opt.test_path = f"{opt.data_root}KAIST_selected_1.3_resize/"
opt.psf_path = f"./datasets/hardware_para/" + opt.psf_name
opt.ff_path = f"./datasets/hardware_para/" + opt.ff_name


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False






