import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches


def imadjust(image, tol=[0.01, 0.99]):
    # img : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 1.

    def adjust_single_channel(img, tol):
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

    if len(image.shape) == 2:
        return adjust_single_channel(image, tol)
    elif len(image.shape) == 3:
        for i in range(3):
            image[:,:,i] = adjust_single_channel(image[:, :, i], tol)
        return image

def visualize_bbxs(image, centers=None, bbxs=None, save=False, adjust_hist=False):
    # adjust the histogram of the image
    if adjust_hist:
        image = imadjust(image)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    if centers is not None:
        ax.plot(centers[:, 0], centers[:, 1], 'b.')
    if bbxs is not None:
        for idx in range(bbxs.shape[0]):
            xmin = bbxs[idx, 0]
            ymin = bbxs[idx, 1]
            width = bbxs[idx, 2] - bbxs[idx, 0]
            height = bbxs[idx, 3] - bbxs[idx, 1]
            ax.add_patch(patches.Rectangle((xmin, ymin), width, height,
                                           edgecolor="blue", fill=False))
    if save:
        fig.savefig('visualized_image.png', bbox_inches='tight')
    else:
        plt.show()
    