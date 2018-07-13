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
        self.channel = cfg.channel
        self.ovrlp = cfg.crop_overlap
        self.image = read_image_from_filenames([cfg.img_1, cfg.img_2, cfg.img_3], to_ubyte=False)

        if cfg.mode == 'test':
            self.centers = []
            self.bbxs = []
            self.scores = []

    def next_crop(self):

        # get image information
        img_rows, img_cols, img_ch = self.image.shape  # img_rows = height , img_cols = width
        crop_size = (self.width, self.height)

        # get each crop
        for i in range(0, img_rows, self.height - self.ovrlp):
            for j in range(0, img_cols, self.width - self.ovrlp):
                # crop the image
                crop_img = self.image[i:i + self.height, j:j + self.width, :]  # create crop image
                # if we were at the edges of the image, zero pad the crop
                if crop_img.shape[:2][::-1] != crop_size:
                    temp = np.copy(crop_img)
                    crop_img = np.zeros((self.height, self.width, 3))
                    crop_img[:temp.shape[0], :temp.shape[1], :] = temp

                yield [j, i], crop_img

    def get_centers(self):
        self.centers = np.empty((self.bbxs.shape[0], 2), dtype=int)
        self.centers[:, 0] = (self.bbxs[:, 0] + self.bbxs[:, 2]) // 2
        self.centers[:, 1] = (self.bbxs[:, 1] + self.bbxs[:, 3]) // 2

        return self.centers


    def non_max_suppression_fast(self, overlapThresh):
        # Malisiewicz et al.
        # if there are no boxes, return an empty list
        boxes = self.bbxs
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")


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