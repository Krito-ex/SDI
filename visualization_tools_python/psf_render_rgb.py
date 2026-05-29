import numpy as np
import matplotlib.pyplot as plt
import os
from utils.spectrumRGB import spectrumRGB
import scipy.io as sio



def psf_render_rgb(cube, wavelengths, save_name, save_rendered_name, single_save=False):
    H, W, num_channel = cube.shape
    sRGB_list = spectrumRGB(wavelengths)

    cube_R = cube * sRGB_list[:,0][np.newaxis, np.newaxis, :]
    cube_G = cube * sRGB_list[:,1][np.newaxis, np.newaxis, :]
    cube_B = cube * sRGB_list[:,2][np.newaxis, np.newaxis, :]

    max = np.max([np.max(cube_R), np.max(cube_G), np.max(cube_B)])
    I_para = 1. / max
    cube_R, cube_G, cube_B = cube_R* I_para, cube_G*I_para, cube_B*I_para

    if single_save == True:
        os.makedirs(save_rendered_name.split('.png')[0], exist_ok=True)
        for i in range(num_channel):
            wavelength = wavelengths[i]
            single_channel_rgb = np.dstack((cube_R[:,:,i], cube_G[:,:,i], cube_B[:,:,i]))
            single_channel_rgb = single_channel_rgb / np.max(single_channel_rgb)

            eps = 1e-12
            floor_db = -28   
            single_channel_rgb = 10*np.log10(single_channel_rgb + eps)
            single_channel_rgb = np.clip(single_channel_rgb, floor_db, 0.0)
            single_channel_rgb = (single_channel_rgb - floor_db) / (0.0 - floor_db) # mappig to 0，1
            plt.imsave(save_rendered_name.split('.png')[0] + '/{:.2f}.png'.format(wavelength), single_channel_rgb)

    R = np.sum(cube_R, axis=2) / num_channel
    G = np.sum(cube_G, axis=2) / num_channel
    B = np.sum(cube_B, axis=2) / num_channel

    rendered_rgb = np.dstack((R, G, B))

                           
    # log
    eps = 1e-12
    floor_db = -28   
    rendered_rgb = 10*np.log10(rendered_rgb + eps)
    rendered_rgb = np.clip(rendered_rgb, floor_db, 0.0)
    rendered_rgb = (rendered_rgb - floor_db) / (0.0 - floor_db) # mappig to 0，1

    
    plt.imsave(save_rendered_name, rendered_rgb)


if __name__ == '__main__':

    data_root = '/home/data1/lvtao/Meta-adis-amax-151/VST-SDI/ADIS_tools_new/PSF_Fab_error/'
    save_root = '/home/data1/lvtao/Meta-adis-amax-151/VST-SDI/ADIS_tools_new/PSF_Fab_error_show_log/'

    if not os.path.exists(save_root):
        os.mkdir(save_root)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        psf = np.load(os.path.join(data_root, filename))
        if psf.shape[1] == 1024:
            psf = psf[:,256:768,256:768]
        psf = psf.transpose(1,2,0)    # [H,W,C]
        psf = np.array(psf, dtype=np.float32)
        psf = psf/ psf.max()
        psf[psf<0] = 0
        psf[psf>1] = 1
        save_name = os.path.join(save_root, filename.split('.npy')[0]+'.png')
        wavelengths = np.linspace(450, 650, 9)
        psf_render_rgb(psf, wavelengths, save_root, save_name, single_save=True)
