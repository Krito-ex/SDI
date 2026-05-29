import numpy as np
from scipy.interpolate import interp1d
from skimage.color import xyz2rgb
from .colorMatchFcn import colorMatchFcn


def spectrumRGB(lambdaIn, *args):
    """
    spectrumRGB   Converts a spectral wavelength to RGB.

    Parameters:
    - lambdaIn: scalar or vector of wavelengths (in nm).
    - varargin: optional color matching function (string).

    Returns:
    - sRGB: RGB values in the sRGB color space.

    Note: This function requires appropriate color matching function data.
    """

    if isinstance(lambdaIn, (list, np.ndarray)):
        lambdaIn = np.array(lambdaIn)
    else:
        lambdaIn = np.array([lambdaIn])

    if lambdaIn.ndim != 1:
        raise ValueError('Input must be a scalar or vector of wavelengths.')

    if len(args) == 0:
        matchingFcn = '1964_full'  # Default color matching function
    elif len(args) == 1:
        matchingFcn = args[0]
    else:
        raise ValueError('Unsupported number of arguments.')

    lambdaMatch, xFcn, yFcn, zFcn = colorMatchFcn(matchingFcn)

    # Interpolate the input wavelength in the color matching functions
    interp_x = interp1d(lambdaMatch, xFcn, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(lambdaMatch, yFcn, kind='linear', fill_value="extrapolate")
    interp_z = interp1d(lambdaMatch, zFcn, kind='linear', fill_value="extrapolate")

    XYZ = np.zeros((len(lambdaIn), 3))
    XYZ[:, 0] = interp_x(lambdaIn)
    XYZ[:, 1] = interp_y(lambdaIn)
    XYZ[:, 2] = interp_z(lambdaIn)

    # Convert XYZ to sRGB
    sRGB = xyz2rgb(XYZ)

    return sRGB
