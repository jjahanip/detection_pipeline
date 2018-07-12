import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', 'train or test')

# Detection pipeline parameters
flags.DEFINE_string('pipeline_config_path', 'training/NeuN/pipeline.config', 'Path to detection config file')
flags.DEFINE_string('trained_checkpoint', 'training/NeuN/model.ckpt-200000', 'Path to trained checkpoint')
flags.DEFINE_string('input_shape', None, 'Comma delimited input shape e.g 300,300,3')

# data
flags.DEFINE_boolean('data_augment', False, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 300, 'Network input height size')
flags.DEFINE_integer('width', 300, 'Network input width size')
flags.DEFINE_integer('depth', None, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 3, 'Network input channel size')
flags.DEFINE_string('img_1', 'data/test/hpc_crop/R2C0_crop_crop.tif', 'image 1 path')
flags.DEFINE_string('img_2', 'data/test/hpc_crop/R2C1_crop_crop.tif', 'image 2 path')
flags.DEFINE_string('img_3', 'data/test/hpc_crop/R2C3_crop_crop.tif', 'image 3 path')
flags.DEFINE_integer('crop_overlap', 25, 'Network input channel size')

# Network paramteres
flags.DEFINE_integer('batch_size', 5, 'training batch size')
flags.DEFINE_float('score_threshold', .4, 'Threshold of score of detection box')
flags.DEFINE_integer('max_proposal', 400, 'maximum proposal per image')


args = tf.app.flags.FLAGS