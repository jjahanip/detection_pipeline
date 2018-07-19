import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', 'train or test')

# Detection pipeline parameters
flags.DEFINE_string('pipeline_config_path', 'training/NeuN/pipeline.config', 'Path to detection config file')
flags.DEFINE_string('trained_checkpoint', 'training/NeuN/model.ckpt-200000', 'Path to trained checkpoint')
flags.DEFINE_string('input_shape', None, 'Comma delimited input shape e.g 300,300,3')

# data
flags.DEFINE_string('data_dir', 'data/test/whole', 'Path to the directory of input data')
flags.DEFINE_string('c1', 'R2C0.tif', 'image 1 path')
flags.DEFINE_string('c2', 'R2C1.tif', 'image 2 path')
flags.DEFINE_string('c3', 'R2C3.tif', 'image 3 path')
flags.DEFINE_integer('height', 300, 'Network input height size - crop large image with this height')
flags.DEFINE_integer('width', 400, 'Network input width size - crop large image with this height')
flags.DEFINE_integer('depth', None, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 2, 'Network input channel size')
flags.DEFINE_boolean('data_augment', False, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('crop_overlap', 100, 'Network input channel size')

# post processing
flags.DEFINE_float('nms_iou', .6, 'intersection over union of bbxs for non max suppression')
flags.DEFINE_integer('close_centers_r', 3, 'Minimum distance between two centers')

# Network paramteres
flags.DEFINE_integer('batch_size', 8, 'training batch size')
flags.DEFINE_float('score_threshold', .5, 'Threshold of score of detection box')
flags.DEFINE_integer('max_proposal', 200, 'maximum proposal per image')


args = tf.app.flags.FLAGS