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


class JNet(object):
    def __init__(self, conf):

        if conf.input_shape is None:
            self.input_shape = (None, None, None, 3)
        else:
            self.input_shape = conf.input_shape
        self.input = None
        self.outputs = None

        self.conf = conf
        self.build_graph()

    def build_graph(self):
        # read pipeline config
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(self.conf.pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge('', pipeline_config)

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

    def test(self):
        data = DataLoader(self.conf)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.conf.trained_checkpoint)
            for batch in data.next_batch():
                return sess.run(self.outputs, feed_dict={self.input: input})


def main():

    # args
    pipeline_config_path = r'training/NeuN/test_300.config'
    trained_checkpoint_prefix = r'training/NeuN/model.ckpt-200000'
    input_shape = None
    PATH_TO_TEST_IMAGES_DIR = r'test_images'

    jNet = JNet(test_config=pipeline_config_path,
                trained_checkpoint=trained_checkpoint_prefix,
                input_shape=input_shape)


    # read image:
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, test_image) for test_image in
                        os.listdir(PATH_TO_TEST_IMAGES_DIR)]
    image_np = read_image_from_filenames(TEST_IMAGE_PATHS, to_ubyte=False).astype(float)
    image_np = image_np / 255
    image_np = image_np[1000:1300, 1000:1300, :]
    # image_np = load_image_into_numpy_array(image)
    image_np = np.expand_dims(image_np, axis=0)

    # run session
    with tf.Session() as sess:
        jNet.saver.restore(sess, 'training/NeuN/model.ckpt-200000')
        out_dict = sess.run(jNet.output_tensors, feed_dict={'image_tensor:0': image_np})
        prop_class_feat = sess.run('SecondStageFeatureExtractor/InceptionResnetV2/Conv2d_7b_1x1/Relu:0',
                                   feed_dict={'image_tensor:0': image_np})

    # visualize
    from lib.image_uitls import visualize_bbxs
    keep_boxes = out_dict["detection_scores"] > .5
    boxes = out_dict["detection_boxes"][keep_boxes, :]
    crop_bbxs = []
    crop_size = [300, 300]
    for i, box in enumerate(boxes):
        box = box.tolist()
        ymin, xmin, ymax, xmax = box

        # for crop visualization
        crop_bbxs.append([xmin * crop_size[0],
                          ymin * crop_size[1],
                          xmax * crop_size[0],
                          ymax * crop_size[1]])
    visualize_bbxs(np.squeeze(image_np / 255), bbxs=np.array(crop_bbxs))

if __name__ == '__main__':
    main()
    print()
