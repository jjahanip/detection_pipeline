# detection_pipeline:

detection_pipeline is a tool for cell detection.

# Dependencies:

* Tensorflow v1.4.* or later!

* [Protobuf](https://github.com/google/protobuf/releases)

# Installation:
  1. Download Protobuf from [here]((https://github.com/google/protobuf/releases)) and run the following command from ```lib``` directory:
  ``` bash
  protoc object_detection/protos/*.proto --python_out=.
  ```
  __*NOTE:__ Make sure the executable folder is added to your path variable:

    - (__for Windows__):
      ```bash
      # From Protoc*/bin
      SET PATH=%PATH%;%cd%
      ```
    
    - (__for Linux and Mac__):
      ```bash
      # From Protoc*/bin
      export PATH=$PATH:`pwd`
      ```

  2. Add object_detection and slim folders to PYTHONPATH.

    - (__for Windows__):
      ```bash
      # From tensorflow/models/research/
      SET PYTHONPATH=%cd%;%cd%\slim
      ```
    
    - (__for Linux and Mac__):
      ```bash
      # From tensorflow/models/research/
      export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
      ```

# Pipeline:

### 1. Create small crops from large images and centers
```bash
python write_crops.py --images_dir=data/input_data --crop_size=300,300 --save_dir=data --adjust_image --visualize=2
```
__NOTE__: Use visualize if you want to see the first "n" crops to make sure everything is right.

### 2. (optional) Fix bounding boxes usign [LabelImg](https://github.com/tzutalin/labelImg)

### 3. Generate tfrecord file from xmls and imgs
```bash
python generate_tfrecord.py --input_dir=data --output_path=data/train.record
```
Create a label map for mapping classes to unique IDs. For example create a ```nucleus_map.pbtxt``` file inside ```data``` folder and add following lines:
```vim
item {
name: "Nucleus"
id: 1
display_name: "Nucleus"
}
```

### 4. Train:
1. Download your pretrained model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and save it in folder ```pretrained_models ```.

2. create a __training__ folder to save the training model and parameters. Inside the __training__ folder copy the .config file from ```lib/object_detection/samples/configs/```.
For example ```faster_rcnn_inception_resnet_v2_atrous_coco.config``` file.

  * edit the following lines:
  ```vim
  num_classes: 1
  fine_tune_checkpoint: pretrained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
  input_path: data/train.record
  label_map_path: data/nucleus_map.pbtxt
  ```
  * and comment the evaluation lines:
  ```vim
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
3. Now you can train your model:
  ```bash
  python train.py --logtostderr --train_dir=training --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_atrous_coco.config
  ```
  
### 5. Export Inference Graph:
Run the following command to export the inference graph for test:
```bash
python export_inference_graph.py --ipnut_type=image_tensor --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_atrous_coco.config --trained_checkpoint_prefix=training/model.ckpt-20000 --output_directory=new_model
```
__NOTE__: Make sure you have all 3 ```.index```,```.meta``` and ```.data``` files for that checkpoint.

### 6. Visualization Test:
1. Create ```test_image``` folder and put some sample images.
2. run ```test.ipynb``` in jupyter notebook.

### 7. Detect cells in large images:
coming soon ...


# Probable Errors:

1. If you faced this error:  
  ```bash
  ImportError: No module named 'object_detection'
  ```

  Add object_detection and slim folders to PYTHONPATH.  
  
  - (__for Windows__):
    ```bash
    # From tensorflow/models/research/
    SET PYTHONPATH=%cd%;%cd%\slim
    ```

  - (__for Linux and Mac__):
    ```bash
    # From tensorflow/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```

2. If your TensorFlow version is  < 1.5 you might have issues with object detection module. Go to commit 196d173 which is compatible with tf 1.4.1:

  ```bash
  # from tensorflwo/models
  git checkout 196d173
  ```

3. If your TensorFlow version is > 1.5 you might have compatibility issue with python3.x. If you faced the following error:
  ```bash
  ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].
  ```
  In ```models/research/object_detection/utils/learning_schedules.py``` lines 167-169, Wrap ```list()``` around the ```range()``` like this:
  ```python
  rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                       list(range(num_boundaries)),
                                        [0] * num_boundaries))
  ```
  
4. If stucked with ```INFO:tensorflow:global_step/sec: 0``` you might have some issues with the ```.record``` data file. Double check your input data file.
