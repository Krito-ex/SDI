import numpy as np
import h5py
from pathlib import Path

def colorMatchFcn(formulary):
    # Initialize variables
    lambda_ = []
    xFcn = []
    yFcn = []
    zFcn = []

    
    root = str(Path(__file__).resolve().parent)


    # Define color matching functions
    if formulary.lower() == 'judd_vos':
        cmf = h5py.File(root + '/cmf/judd_vos.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == 'judd':
        cmf = h5py.File(root + '/cmf/judd.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == 'cie_1931':
        cmf = h5py.File(root + '/cmf/cie_1931.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == 'stiles_2':
        cmf = h5py.File(root + '/cmf/stiles_2.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == 'stiles_10':
        cmf = h5py.File(root + '/cmf/stiles_10.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == 'cie_1964':
        cmf = h5py.File(root + '/cmf/cie_1964.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == '1931_full':
        cmf = h5py.File(root + '/cmf/1931_full.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    elif formulary.lower() == '1964_full':
        cmf = h5py.File(root + '/cmf/1964_full.mat', 'r')['cmf']
        cmf = np.array(cmf, dtype=np.float64).transpose()
    else:
        raise ValueError('Unrecognized color match function.')

    # Extracting data
    lambda_ = cmf[:, 0]
    xFcn = cmf[:, 1]
    yFcn = cmf[:, 2]
    zFcn = cmf[:, 3]

    return lambda_, xFcn, yFcn, zFcn

if __name__ == '__main__':
    # root = Path(__file__).resolve().parent
    # print(root)
    # Example usage:
    lambda_, xFcn, yFcn, zFcn = colorMatchFcn('judd_vos')
    print("Wavelengths:", lambda_)
    print("X function:", xFcn)
    print("Y function:", yFcn)
    print("Z function:", zFcn)
