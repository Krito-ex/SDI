import numpy as np
import matplotlib.pyplot as plt
import os
from utils.spectrumRGB import spectrumRGB
import scipy.io as sio


def hsi_render_rgb(cube, wavelengths, save_name, save_rendered_name1, save_rendered_name2, single_save=False):
    H, W, num_channel = cube.shape
    sRGB_list = spectrumRGB(wavelengths)

    cube_R = cube * sRGB_list[:,0][np.newaxis, np.newaxis, :]
    cube_G = cube * sRGB_list[:,1][np.newaxis, np.newaxis, :]
    cube_B = cube * sRGB_list[:,2][np.newaxis, np.newaxis, :]

    max = np.max([np.max(cube_R), np.max(cube_G), np.max(cube_B)])
    I_para = 1. / max
    cube_R, cube_G, cube_B = cube_R* I_para, cube_G*I_para, cube_B*I_para

    if single_save == True:
        os.makedirs(save_name, exist_ok=True)
        for i in range(num_channel):
            print(i)
            wavelength = wavelengths[i]
            single_channel_rgb = np.dstack((cube_R[:,:,i], cube_G[:,:,i], cube_B[:,:,i]))
            single_channel_rgb = single_channel_rgb / np.max(single_channel_rgb)
            # plt.imshow(single_channel_rgb)
            # plt.show()
            plt.imsave(f'{save_name}/' + '{:.2f}.png'.format(wavelength), single_channel_rgb)

    R = np.sum(cube_R, axis=2) / num_channel
    G = np.sum(cube_G, axis=2) / num_channel
    B = np.sum(cube_B, axis=2) / num_channel
    # R, G, B = R/np.max(R), G/np.max(G), B/np.max(B)
    rendered_rgb = np.dstack((R, G, B))
    # plt.imshow(rendered_rgb)
    # plt.show()
    plt.imsave(save_rendered_name1, rendered_rgb)
    R, G, B = R / np.max(R), G / np.max(G), B / np.max(B)
    rendered_blance_rgb = np.dstack((R, G, B))
    rendered_blance_rgb = rendered_blance_rgb / np.max(rendered_blance_rgb)
    plt.imsave(save_rendered_name2, rendered_blance_rgb)



if __name__ == '__main__':

    data_root = './real/test_SDI_ADIS_Normal/51S5C_Rreal_result_HSFAUTVfinal_3stg_final'

    filenames = os.listdir(data_root)
    filenames.sort()
    save_root = data_root + '_show'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for filename in filenames:
        # if filename.endswith('.mat') and 'SPATIAL' in filename:
        if filename.endswith('.mat'):
            data_path = os.path.join(data_root, filename)



            cube = sio.loadmat(data_path)['pred']
            cube = np.array(cube, dtype=np.float32).transpose(1, 2, 0)  # [H,W,C]
            [h,w,c] = cube.shape

            cube[cube<0] = 0.
            cube[cube>1] = 1.

            # Define the directory to save results
            save_name = os.path.join(save_root, filename.split('.mat')[0])

            save_rendered_name1 = os.path.join(save_root, filename.split('.mat')[0]+'_rgb.png')
            save_rendered_name2 = os.path.join(save_root, filename.split('.mat')[0] + '_rgb_balance.png')

            # wavelengths = np.linspace(450, 700, 30)
            wavelengths = np.linspace(450, 650, (28 + 1)) + (650 - 450) / (2 * 28)
            wavelengths = wavelengths[0:28]
            hsi_render_rgb(cube, wavelengths, save_name, save_rendered_name1, save_rendered_name2, single_save=True)