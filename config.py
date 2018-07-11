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

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')


# For training
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_integer('val_batch_size', 100, 'validation batch size')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Directories
flags.DEFINE_string('run_name', 'run2', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('savedir', './Results/result/', 'Results saving directory')

flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

# network architecture
flags.DEFINE_integer('num_cls', 10, 'Number of output classes')
flags.DEFINE_integer('digit_caps_dim', 16, 'Dimension of the DigitCaps')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')



args = tf.app.flags.FLAGS