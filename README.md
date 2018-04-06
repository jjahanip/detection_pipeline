# detection_pipeline:

detection_pipeline is a tool for cell detection.

# Pipeline:

1. Create small crops from large image and centers
```bash
python write_crops.py \
--images_dir=data/input_data \
--crop_size=300,300 \
--save_dir=data \
--adjust_image \
--visualize=2
```
__NOTE__: Use visualize if you want to see the first "n" crops to make sure everything is right.

2. (optional) Fix bounding boxes usign [LabelImg](https://github.com/tzutalin/labelImg)

3. Generate tfrecord file from xmls and imgs
  ```bash
  python generate_tfrecord.py \ 
  --input_dir=data \
  --output_path=data/train.record
  ```
  Create a label map for mapping classes to unique IDs. For example create a ```nucleus_map.pbtxt``` file inside ```data`` folder and add following lines:
  ```txt
  item {
  name: "Nucleus"
  id: 1
  display_name: "Nucleus"
  }
  ```

4. Train: 
  1. Locate your object_detection folder and copy the train.py file to the ```detection_pipeline``` directory.

  2. Download your pretrained model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and save it in folder ```pretrained_models ```.

  3. create a __training__ folder to save the training model and parameters. Inside the __training__ folder copy the .config file from ```/tensorflow/models/research/object_detection/samples/configs/```.
  For example ```faster_rcnn_inception_resnet_v2_atrous_coco.config``` file.

    edit the following lines:

    ```txt
    num_classes: 1
    fine_tune_checkpoint: pretrained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
    input_path: data/train.record
    label_map_path: data/nucleus_map.pbtxt
    ```
    and comment the evaluation lines:
    ```txt
    # eval_config: {
    #   num_examples: 8000
    #   # Note: The below line limits the evaluation process to 10 evaluations.
    #   # Remove the below line to evaluate indefinitely.
    #   max_evals: 10
    # }
    # 
    # eval_input_reader: {
    #   tf_record_input_reader {
    #   input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
    # }
    # label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
    # shuffle: false
    # num_readers: 1
    # }
    ```
  
    Now you can train your model:
    ```bash
    python train.py \
    --logtostderr \
    --train_dir=training \
    --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_atrous_coco.config
    ```

### Dependencies

* Tensorflow
* Object detection toolkit (download from [here](https://github.com/tensorflow/models) and install from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

* * (__for windows__): Install protofub from [here](https://github.com/google/protobuf/releases).
* * Add object_detection and slim folders to PYTHONPATH.

    (__for Windows__):
```bash
# From tensorflow/models/research/
SET PYTHONPATH=%cd%;%cd%\slim
```
    (__for Linux and Mac__):
```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```


### Probable Errors:
* If your TensorFlow version is  < 1.5 you might have issues with object detection module. Go to commit 196d173 which is compatible with tf 1.4.1:
```bash
# from tensorflwo/models
git checkout 196d173
```


 * If your TensorFlow version is > 1.5 you might have compatibility issue with python3.x. If you faced the following error:
```bash
ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].
```
In ```models/research/object_detection/utils/learning_schedules.py``` lines 167-169, Wrap ```list()``` around the ```range()``` like this:
```python
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),
                                      [0] * num_boundaries))
```
[Reference](https://github.com/tensorflow/models/issues/3705)

* If stucked with ```INFO:tensorflow:global_step/sec: 0``` you might have some issues with the ```.record``` data file. Double check your input data file.