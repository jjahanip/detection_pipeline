import os
import pandas as pd
import random
import scipy
import numpy as np
import scipy.spatial as spatial
import warnings
import skimage
from skimage import exposure
import xml.etree.ElementTree as ET
import itertools
from lib.image_uitls import read_image_from_filenames, visualize_bbxs
from lib.segmentation import GenerateBBoxfromSeeds
from lib.ops import write_xml
import progressbar

class DataLoader(object):

    def __init__(self, config):

        # read all images
        image_filenames = []
        for i in range(config.channel):
            filename = getattr(config, 'c{}'.format(i+1))
            image_filenames.append(os.path.join(config.data_dir, filename))
        self.image = read_image_from_filenames(image_filenames, to_ubyte=False)

        self.height = config.height
        self.width = config.width
        self.channel = config.channel
        self.ovrlp = config.crop_overlap

        # read centers if exist
        if os.path.isfile(os.path.join(config.data_dir, 'centers.txt')):
            centers_table = pd.read_csv(os.path.join(config.data_dir, 'centers.txt'), sep='\t')
            self.centers = centers_table[['centriod_x', 'centriod_y']].values
        else:
            self._centers = None

        # read bbxs if exist
        if os.path.isfile(os.path.join(config.data_dir, 'bbxs.txt')):
            bbxs_table = pd.read_csv(os.path.join(config.data_dir, 'bbxs.txt'), sep='\t')
            self.bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values
        else:
            self._bbxs = None

        self._scores = None


    @property
    def centers(self):
        return self._centers

    # @centers.setter
    # def centers(self, value):
    #     if self._bbxs is None:
    #         self._centers = value
    #     else:
    #         print('Data has bbxs. Over-writting centers...')
    #         self._centers = value

    @staticmethod
    def get_centers(bbxs):
        centers = np.empty((bbxs.shape[0], 2), dtype=int)
        centers[:, 0] = (bbxs[:, 0] + bbxs[:, 2]) // 2
        centers[:, 1] = (bbxs[:, 1] + bbxs[:, 3]) // 2
        return centers

    @property
    def bbxs(self):
        return self._bbxs

    @bbxs.setter
    def bbxs(self, value):
        self._bbxs = value
        self._centers = self.get_centers(value)

    def save_bbxs(self, filename):
        # create a column for unique IDs
        ids = np.arange(1, self._bbxs.shape[0] + 1)

        # create numpy array for the table
        table = np.hstack((ids, self._centers, self._bbxs))

        fmt = '\t'.join(['%d'] * table.shape[1])
        hdr = '\t'.join(['ID'] + ['centroid_x'] + ['centroid_y'] + ['xmin'] + ['ymin'] + ['xmax'] + ['ymax'])
        cmts = ''

        np.savetxt(filename, table, fmt=fmt, header=hdr, comments=cmts)

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, value):
        self._scores = value

    def next_crop(self):

        # get image information
        img_rows, img_cols, img_ch = self.image.shape  # img_rows = height , img_cols = width
        max_bar = (img_rows // (self.height - self.ovrlp) + 1) * (img_cols // (self.width - self.ovrlp) + 1)

        bar = progressbar.ProgressBar(max_value=max_bar)

        bar.start()
        bar_idx = 1
        # get each crop
        for i in range(0, img_rows, self.height - self.ovrlp):
            for j in range(0, img_cols, self.width - self.ovrlp):

                # update bar
                bar.update(bar_idx)

                # temporary store the values of crop
                temp = self.image[i:i + self.height, j:j + self.width, :]

                # create new array to copy temporary stored values
                crop_img = np.zeros((self.height, self.width, self.image.shape[-1]), dtype=self.image.dtype)
                crop_img[:temp.shape[0], :temp.shape[1], :] = temp

                yield [j, i], crop_img
                bar_idx += 1
        bar.finish()

    @staticmethod
    def remove_close_centers(centers, scores=None, radius=3):
        """ returns array of True and Flase for centers to keep(True) remove(False) """

        # get groups of centers
        tree = spatial.cKDTree(centers)
        groups = tree.query_ball_point(centers, radius)

        # remove isolated centers
        groups = [group for group in groups if len(group) > 1]

        # if no groups, return all True
        if len(groups) == 0:
            return np.array([True] * centers.shape[0])

        # remove duplicated groups
        groups.sort()
        groups = list(groups for groups,_ in itertools.groupby(groups))

        # remove the center with highest probability and add to to_be_removed list
        to_be_removed = []
        for i, group in enumerate(groups):
            if scores is not None:
                max_idx = np.argmax(scores[group])
            else:
                # if we don't have the scores remove the first object
                max_idx = 0
            to_be_removed.append(np.delete(groups[i], max_idx))

        # find index of centers to be removed
        to_be_removed = np.unique(list(itertools.chain.from_iterable(to_be_removed)))

        # update bbxs, centers and scores
        # self.bbxs = np.delete(self._bbxs, to_be_removed, axis=0)

        # if self._scores is not None:
        #     self.scores = np.delete(self._scores, to_be_removed, axis=0)

        # return invert of to_be_removed = to_keep
        return np.isin(np.arange(centers.shape[0]), to_be_removed, invert=True)

    def write_crops(self, save_folder, adjust_hist=False):

        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        # check for subdirectories
        dir_list = os.listdir(save_folder)
        if 'imgs' not in dir_list:
            os.mkdir(os.path.join(save_folder, 'imgs'))
        if 'xmls' not in dir_list:
            os.mkdir(os.path.join(save_folder, 'xmls'))

        crop_gen = self.next_crop()
        idx = 1
        while True:
            try:
                [j, i], crop_image = next(crop_gen)
            except StopIteration:
                break

            if self._bbxs is not None:
                # find cells that center is in the crop
                crop_idx = np.where(((self.centers[:, 0] > j) &
                                     (self.centers[:, 0] < j + self.width) &
                                     (self.centers[:, 1] > i) &
                                     (self.centers[:, 1] < i + self.height)),
                                    True, False)

                if not np.any(crop_idx):  # if no cell in the crop, SKIP
                    continue

            crop_centers = self.centers[crop_idx, :]

            # shift the x & y values based on crop size
            crop_centers[:, 0] = crop_centers[:, 0] - j
            crop_centers[:, 1] = crop_centers[:, 1] - i

            # if user provides the bounding boxes
            if self._bbxs is not None:
                # extract bbxs in the crop
                crop_bbxs = self.bbxs[crop_idx, :]
                # shift the x & y values based on crop size
                crop_bbxs[:, [0, 2]] = crop_bbxs[:, [0, 2]] - j
                crop_bbxs[:, [1, 3]] = crop_bbxs[:, [1, 3]] - i
            else:
                # generate bounding boxes using segmentation
                dapi = np.copy(crop_image[:, :, 0])
                # for 16bit images only.
                # TODO: general form for all types
                dapi = exposure.rescale_intensity((dapi // 256).astype('uint8'),
                                                  in_range='image', out_range='dtype')
                crop_bbxs = GenerateBBoxfromSeeds(dapi, crop_centers)

            # find truncated objects in crop
            crop_truncated = np.where(((crop_bbxs[:, 0] < 0) |
                                       (crop_bbxs[:, 1] < 0) |
                                       (crop_bbxs[:, 2] > self.width) |
                                       (crop_bbxs[:, 3] > self.height)), True, False)
            # clip truncated objects
            if np.any(crop_truncated):
                crop_bbxs[:, [0, 2]] = np.clip(crop_bbxs[:, [0, 2]], 1, self.width - 1)
                crop_bbxs[:, [1, 3]] = np.clip(crop_bbxs[:, [1, 3]], 1, self.height - 1)

            # save image and xml:
            filename = '{:05}'.format(idx)
            if adjust_hist:
                # for 16bit images only.
                # TODO: general form for all types
                crop_image = exposure.rescale_intensity((crop_image // 256).astype('uint8'),
                                                        in_range='image', out_range='dtype')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(os.path.join(save_folder, 'imgs', filename + '.jpeg'),
                                  np.squeeze(crop_image))  # save the image

            # write bounding boxes in xml file
            labels = ['Nucleus'] * crop_bbxs.shape[0]
            truncated = crop_truncated * 1
            write_xml(os.path.join(save_folder, 'xmls', filename + '.xml'), corner=[j, i],
                      bboxes=crop_bbxs, labels=labels, truncated=truncated,
                      image_size=[self.width, self.height, self.channel])

            # visualize_bbxs(crop_image, bbxs=crop_bbxs, centers=crop_centers)
            idx += 1

        print('{} images in: {}'.format(idx - 1, save_folder + '/imgs'))
        print('{} xmls in: {}'.format(idx - 1, save_folder + '/xmls'))

    def update_xmls(self, xml_dir, centers_radius=3):

        to_be_deleted = []
        to_be_added = []
        for filename in os.listdir(xml_dir):

            # read file
            tree = ET.parse(os.path.join(xml_dir, filename))
            source = tree.find('source')
            corner = [int(c) for c in source.find('corner').text.split(',')]

            size = tree.find('size')
            crop_width = int(size.find('width').text)
            crop_height = int(size.find('height').text)

            # find objects in center (not close to the edge) of crop to be deleted from self.bbxs
            to_be_deleted_crop = np.where((self._bbxs[:, 0] > corner[0] + 50) &
                                          (self._bbxs[:, 1] > corner[1] + 50) &
                                          (self._bbxs[:, 2] < corner[0] + crop_width - 50) &
                                          (self._bbxs[:, 3] < corner[1] + crop_height - 50))[0]
            to_be_deleted.append(to_be_deleted_crop) if len(to_be_deleted_crop) > 0 else None

            # extract bbxs from xml file
            for i, Obj in enumerate(tree.findall('object')):  # take the current animal
                bndbox = Obj.find('bndbox')
                box = np.array([int(bndbox.find('xmin').text),
                                int(bndbox.find('ymin').text),
                                int(bndbox.find('xmax').text),
                                int(bndbox.find('ymax').text)])

                # if box was inside the crop (not close to the edge) to be added to self.bbxs
                if box[0] >= 50 and box[1] >= 50 and box[2] <= crop_width - 50 and box[3] <= crop_height - 50:
                    box = box + np.array([corner[0], corner[1], corner[0], corner[1]])
                    to_be_added.append(box)


        # update the bbxs
        to_be_deleted = [item for sublist in to_be_deleted for item in sublist]
        self.bbxs = np.delete(self.bbxs, to_be_deleted, axis=0)
        self.bbxs = np.vstack([self._bbxs, to_be_added])
        self.bbxs = np.unique(self._bbxs)

    def nms(self, overlapThresh):
        # non_max_suppression_fast
        # Malisiewicz et al.
        # if there are no boxes, return an empty list
        boxes = self._bbxs
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
        self.bbxs = boxes[pick].astype("int")
        self.scores = self._scores[pick]

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        shuffled_x = self.x_train[permutation, :, :, :]
        shuffled_y = self.y_train[permutation]
        return shuffled_x, shuffled_y

    @staticmethod
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

if __name__ == '__main__':

    from config import args
    from lib.image_uitls import bbxs_image
    data = DataLoader(args)
    # data.write_crops(save_folder='data/test/whole', adjust_hist=True)
    # bbxs_image('data/test/whole/old_bbxs.tif', data.bbxs, data.image.shape[:2][::-1])

    data.update_xmls(xml_dir='data/test/whole/xmls', centers_radius=4)
    bbxs_image('data/test/whole/new_bbxs.tif', data.bbxs, data.image.shape[:2][::-1])

    # TODO: add bbxs.txt or centers.txt as arg in config file
    a = 1