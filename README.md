## Pipeline:

1) create small crops from large image and centers -> write_crops.py

2) (optional) fix bounding boxes usign LabelImg

3) generate tfrecord file from xmls and imgs -> generate_tfrecord.py

4) train
```bash
python train.py \
--logtostderr \
--train_dir=/train \
--pipeline_config_path=/train/faster_rcnn_inception_resnet_v2_atrous_coco.config
```

### Dependencies

* Tensorflow
* Object detection toolkit (download from [here](https://github.com/tensorflow/models) and install from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

(__for windows__): Install protofub from [here](https://github.com/google/protobuf/releases)
Add object_detection and slim folders to PYTHONPATH
``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

for tf <1.5 go to commit 196d173 (this is compatible with tf 1.4.1)

### Probable error:

 If you faced the following error:
```bash
ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].
```

Solve it from [here](https://github.com/tensorflow/models/issues/3705)

### Prepare data for train:

 create a __train__ folder to save the train model and parameters. Inside the __train__ folder copy the .config file from /tensorflow/models/research/object_detection/samples/configs/

For example for faster_rcnn_inception_resnet_v2_atrous_coco.config file: correct the following lines:

num_classes: 1

fine_tune_checkpoint: /path/to/downloded/pretrained/model/from/[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) (pretrained_models)

input_path: /path/to/.record __file__ (__data__)

label_map_path: /path/to/map/file/.pbtxt __file__ (__data__)