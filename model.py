import os
import sys
sys.path.append('lib')
sys.path.append('lib/slim')

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields

from DataLoader import DataLoader

from lib.image_uitls import visualize_bbxs, bbxs_image
import progressbar


class JNet(object):
    def __init__(self, conf):

        if conf.input_shape is None:
            self.input_shape = (None, None, None, 3)
        else:
            self.input_shape = conf.input_shape


        self.conf = conf

        self.input = None
        self.outputs = None

        self.build_graph()

    def build_graph(self):
        # read pipeline config
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(self.conf.pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge('', pipeline_config)

        # check to make sure
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = self.conf.height
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = self.conf.width

        pipeline_config.model.faster_rcnn.first_stage_max_proposals = self.conf.max_proposal
        pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = self.conf.max_proposal
        pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = self.conf.max_proposal

        if self.conf.mode == 'test':
            detection_model = model_builder.build(pipeline_config.model, is_training=False)
            self.build_test_graph(detection_model)

        self.saver = tf.train.Saver()


    def build_test_graph(self, detection_model):
        self.input = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
        preprocessed_inputs, true_image_shapes = detection_model.preprocess(self.input)
        output_tensors = detection_model.predict(preprocessed_inputs, true_image_shapes)
        postprocessed_tensors = detection_model.postprocess(output_tensors, true_image_shapes)

        detection_fields = fields.DetectionResultFields
        label_id_offset = 1
        boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
        scores = postprocessed_tensors.get(detection_fields.detection_scores)
        classes = postprocessed_tensors.get(
            detection_fields.detection_classes) + label_id_offset
        masks = postprocessed_tensors.get(detection_fields.detection_masks)
        num_detections = postprocessed_tensors.get(detection_fields.num_detections)
        outputs = {}
        outputs[detection_fields.detection_boxes] = tf.identity(
            boxes, name=detection_fields.detection_boxes)
        outputs[detection_fields.detection_scores] = tf.identity(
            scores, name=detection_fields.detection_scores)
        outputs[detection_fields.detection_classes] = tf.identity(
            classes, name=detection_fields.detection_classes)
        outputs[detection_fields.num_detections] = tf.identity(
            num_detections, name=detection_fields.num_detections)
        if masks is not None:
            outputs[detection_fields.detection_masks] = tf.identity(
                masks, name=detection_fields.detection_masks)
        for output_key in outputs:
            tf.add_to_collection('inference_op', outputs[output_key])

        self.outputs = outputs

    def safe_run(self, sess, feed_dict=None, output_tensor=None):

        try:
            out_dict = sess.run(output_tensor, feed_dict=feed_dict)
        except Exception as e:
            if type(e).__name__ == 'ResourceExhaustedError':
                print('Ran out of memory !')
                print('decrease the batch size')
                sys.exit(-1)
            else:
                print('Error in running session:')
                print(e.message)
                sys.exit(-1)

        return out_dict

    def test(self):
        data = DataLoader(self.conf)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.conf.trained_checkpoint)

            max_bar = np.prod(data.image.shape[:2] // (np.array([data.height, data.width]) - data.ovrlp) + 1)\
                      // data.batch_size + 1
            bar = progressbar.ProgressBar(max_value=max_bar)
            crop_gen = data.next_crop()
            iterate = True
            while iterate:
                bar.update(bar.value + 1)

                # make np arrays for generator
                batch_x = np.empty(shape=(data.batch_size, data.height, data.width, data.channel))
                corner = np.empty(shape=(data.batch_size, 2))

                try:
                    for i in range(data.batch_size):
                        corner[i, :], batch_x[i, :, :, :] = next(crop_gen)
                except StopIteration:
                    iterate = False

                # temp
                batch_x = batch_x / 256

                out_dict = self.safe_run(sess, feed_dict={self.input: batch_x}, output_tensor=self.outputs)

                for i in range(data.batch_size):
                    keep_boxes = out_dict["detection_scores"][i, :] > self.conf.score_threshold

                    if not np.any(keep_boxes):
                        continue

                    box = out_dict["detection_boxes"][i, :][keep_boxes]
                    box = box[:, [1, 0, 3, 2]]      # reformat to: xmin, ymin, xmax, ymax
                    # rescale from [0-1] to the crop size
                    box[:, [0, 2]] = box[:, [0, 2]] * self.conf.width
                    box[:, [1, 3]] = box[:, [1, 3]] * self.conf.height

                    # remove very large bounding boxes
                    box = box[(box[:, 2] - box[:, 0] < 100) | (box[:, 3] - box[:, 1] < 100), :]
                    if box.size == 0:    # if no bounding box after removing large ones
                        continue

                    # visualize_bbxs(batch_x[i, :, :, :].astype('uint8'), bbxs=box, adjust_hist=True)

                    box[:, [0, 2]] += corner[i][0]
                    box[:, [1, 3]] += corner[i][1]

                    data.bbxs.append(box.astype(int))

        # to be added: non-max suppression
        # to be added: rotate crop
        data.bbxs = np.concatenate(data.bbxs)
        bbxs_image('data/test/hpc_crop/bbxs_oop.tif', data.bbxs, (6000, 4000), color='red')
