import os
import sys
sys.path.append('lib')
sys.path.append('lib/slim')

import numpy as np
import tensorflow as tf

from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields

from temp_exporter import _build_detection_graph
from PIL import Image
from lib.image_uitls import read_image_from_filenames


# class JNet(faster_rcnn_inception_resnet_v2_feature_extractor):
class JNet():

    def __init__(self,
                 test_config,
                 trained_checkpoint,
                 input_shape):

        self.test_config = test_config
        self.input_shape = input_shape
        self.trained_checkpoint = trained_checkpoint

        self.input_tensor = None
        self.output_tensors = None
        self.saver = None

        self.build_graph()

    def get_model(self):
        # read pipeline config
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(self.test_config, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge('', pipeline_config)

        return model_builder.build(pipeline_config.model, is_training=False)

    def get_input_tensor(self):
        if self.input_shape is None:
            self.input_shape = (None, None, None, 3)
        return tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='image_tensor')

    def get_output_tensors_from_input(self, input_tensor, detection_model):
        preprocessed_inputs, true_image_shapes = detection_model.preprocess(input_tensor)
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

        return outputs

    def build_graph(self):

        detection_model = self.get_model()

        self.input_tensor = self.get_input_tensor()
        self.output_tensors = self.get_output_tensors_from_input(self.input_tensor, detection_model)

        self.saver = tf.train.Saver()


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
