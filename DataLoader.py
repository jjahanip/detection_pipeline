import random
import scipy
import numpy as np
import progressbar

from lib.image_uitls import read_image_from_filenames

class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        self.height = cfg.height
        self.width = cfg.width

        self.image = read_image_from_filenames([cfg.img_1, cfg.img_2, cfg.img_3], to_ubyte=False)

    def next_batch(self):

        # get image information
        img_rows, img_cols, img_ch = self.image.shape  # img_rows = height , img_cols = width
        crop_size = (self.width, self.height)
        # overlap between crops
        ovrlp = 50
        crop_count = 0

        max_bar = (img_rows // (self.height - ovrlp) + 1) * (img_cols // (self.width - ovrlp) + 1)
        with progressbar.ProgressBar(max_value=max_bar) as bar:
            # get each crop
            for i in range(0, img_rows, self.height - ovrlp):
                for j in range(0, img_cols, self.width - ovrlp):
                    bar.update(crop_count)
                    # crop the image
                    crop_img = self.image[i:i + self.height, j:j + self.width, :]  # create crop image
                    # if we were at the edges of the image, zero pad the crop
                    if crop_img.shape[:2][::-1] != crop_size:
                        temp = np.copy(crop_img)
                        crop_img = np.zeros((self.height, self.width, 3))
                        crop_img[:temp.shape[0], :temp.shape[1], :] = temp

                    crop_count = crop_count + 1

                    yield [j, i], crop_img

        # if self.augment:
        #     x = random_rotation_2d(x, self.cfg.max_angle)

        # return x, y

    def get_validation(self):
        x_valid, y_valid = self.mnist.validation.images, self.mnist.validation.labels
        return x_valid, y_valid

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        shuffled_x = self.x_train[permutation, :, :, :]
        shuffled_y = self.y_train[permutation]
        return shuffled_x, shuffled_y


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)