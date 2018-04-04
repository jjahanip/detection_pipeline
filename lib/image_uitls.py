import numpy as np


def imadjust(img, tol=[0.01, 0.99]):
    # img : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 1.

    assert len(img.shape) == 2, 'Input image should be 2-dims'

    if img.dtype == 'uint8':
        nbins = 255
    elif img.dtype == 'uint16':
        nbins = 65535

    N = np.histogram(img, bins=nbins, range=[0, nbins])      # get histogram of image
    cdf = np.cumsum(N[0]) / np.sum(N[0])                     # calculate cdf of image
    ilow = np.argmax(cdf > tol[0]) / nbins                   # get lowest value of cdf (normalized)
    ihigh = np.argmax(cdf >= tol[1]) / nbins                 # get heights value of cdf (normalized)

    lut = np.linspace(0, 1, num=nbins)                       # create convert map of values
    lut[lut <= ilow] = ilow                                  # make sure they are larger than lowest value
    lut[lut >= ihigh] = ihigh                                # make sure they are smaller than largest value
    lut = (lut - ilow) / (ihigh - ilow)                      # normalize between 0 and 1
    lut = np.round(lut * nbins).astype(img.dtype)            # convert to the original image's type

    img_out = np.array([[lut[i] for i in row] for row in img])  # convert input image values based on conversion list

    return img_out
