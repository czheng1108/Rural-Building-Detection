# A RetinaNet-based Approach for Rural Building Detecting in Remote Sensing Imagery

## Introduction
Accurate and rapid detection of rural buildings can effectively prevent illegal construction in rural areas. 
The detection of buildings under construction can efficiently reduce the cost of illicit construction 
management. However, rural buildings under construction account for a small proportion and characteristics 
are more complex than common buildings. Therefore, accurately and quickly acquire the information of
buildings under construction is the key to detect rural buildings. In this paper, 
a RetinaNet-based approach to accurately and speedily find rural buildings is proposed. 
Firstly, the adaptive weight channel attention network is utilized to optimize the building features 
extracted from UAV images. Then the spatial pyramid network for feature fusion is used to realize the 
fusion of features. After that, a new dataset is created, including various objects with high-quality 
annotation in  1590  images with a resolution of 1000 x 1000. Experiments show that our average 
detection speed is 23.18 frames per second, the accuracy of common buildings is 88.2%, the accuracy of 
buildings under construction is 73.0\%, and the average accuracy is 80.6%. In the three evaluation standard, 
the proposed RetinaNet-based method outperforms than RetinaNet improved 2%, 41.1%, and 21.6%, respectively. 
For more details, please refer to our [paper]().

## Installation
Use `git clone https://github.com/czheng1108/Rural-Building-Detection.git`
to clone the project locally, then add the [dataset](https://pan.baidu.com/s/10X0TbmNiFtXdY1OAHzjU8A) to 
the directory `data` and the [pre-training weights](https://pan.baidu.com/s/12VRFScdWlIsErsgHJ5eFzg) to 
the directory `pre_training_weights`. The images of testset can download from [here](https://pan.baidu.com/s/19qZL0F6uI1MMzznEn0F26A). 
For training of other models, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection).
We recommend using soft link to save dataset, pre-training weights, training logs, etc.

## Training
First you need to use `python setup.py build_ext --inplace` to compile the project. 
Then, you can use the following commands for training:

    python tools/train.py --device 'your device' 
                    --datapath 'your path of dataset' 
                    --num-classes 'number of detected categories (excluding background)'
                    --output_dir 'path where to save log' 
                    --resume 'pre-training weights' 
                    --start-epoch 'start epoch'
                    --epoch 'number of total epochs to run' 
                    --batch_size 'batch size when training'
    
such as:

    python tools/train.py --device 'cuda:0' \
                    --datapath './data' \
                    --num-classes 2 \
                    --output_dir './save_weights' \
                    --resume './pre_training_weights/your_weights.pth' \
                    --epoch 100 \
                    --batch_size 10

You can also set optional parameters by modifying the content in the file of [configs](./config/configs.py). 
After that, you only need to use `python tools/train.py` to training your model.
In addition, we also provide two powerful scripts [validation.py](tools/validation.py) and [predict.py](tools/predict.py).
You can use `validation.py` to verify the accuracy of your model, and use `predict.py`
to detect the buildings of your images. In addition, we provide many useful scripts, 
which are placed in the [gifts](./gifts/README.md) folder.


## Methods(Extraction Code：1234) 

| Method | Backbone | SPN | ATT | AP<sub>c</sub> | AP<sub>b</sub>| AP<sub>50</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | Link |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| SSD300     | VGG-16    | F | F | 83.2 | 49.2 | 66.2 | 1.0  | 21.9 | 52.2 | [Baidu](https://pan.baidu.com/s/1tyT64kEoPPEHhJjvhH9SLg)|
| SSD512     | VGG-16    | F | F | 88.8 | 58.6 | 73.7 | 27.1 | 30.2 | 57.7 | [Baidu](https://pan.baidu.com/s/1WrvqGnXdTt2x_mcgK2-5aw)|
| RetinaNet  | ResNet-50 | F | F | 86.2 | 31.9 | 59.0 | 10.5 | 27.4 | 49.7 | [Baidu](https://pan.baidu.com/s/1-AdcTzlevXHNWtZ9YvtIZw)|
| RetinaNet  | ResNet-101| F | F | 88.0 | 58.2 | 73.1 | 5.6  | 37.2 | 58.2 | [Baidu](https://pan.baidu.com/s/1wV7n94_K_Gib5vK1fUBmEg)|
| YOLOv3     | ResNet-50 | F | F | 82.9 | 40.3 | 61.6 | 7.7  | 24.1 | 51.7 | [Baidu](https://pan.baidu.com/s/13QrfclodpdaQWlU1Zof0Vg)|
| Atss       | ResNet-50 | F | F | 88.8 | 65.0 | 76.9 | 19.2 | 37.3 | 62.2 | [Baidu](https://pan.baidu.com/s/1x41LbkD-YjVTCMmhbCgYtQ)|
| Fcos       | ResNet-50 | F | F | 87.2 | 47.2 | 67.2 | 7.1  | 32.0 | 49.3 | - |
| Faster RCNN| ResNet-50 | F | F | 88.2 | 58.9 | 73.6 | 24.5 | 43.9 | 57.9 | [Baidu](https://pan.baidu.com/s/1vng-GUhu-WTOWeKBzK5pmw)|
| Grid RCNN  | ResNet-50 | F | F | 89.1 | 61.6 | 75.4 | 40.6 | 45.1 | 61.7 | [Baidu](https://pan.baidu.com/s/1O-0i3Z0amDmL7vP6o8aDGA)|
| Vfnet      | ResNet-50 | F | F | 88.9 | 26.0 | 57.4 | 9.0  | 28.6 | 47.6 | -|
| TridentNet | ResNet-50 | F | F | 87.7 | 49.9 | 68.8 | 32.9 | 32.7 | 52.3 | [Baidu](https://pan.baidu.com/s/1q53Mh2lAOXDyA1f2RwMO4g)|
| Our        | ResNet-50 | F | F | 87.9 | 61.9 | 74.9 | 19.9 | 37.2 | 61.3 | [Baidu](https://pan.baidu.com/s/1yWwD4QW-6bYzeVzWtEMVUw) / [Details](https://pan.baidu.com/s/1jqjGHDZXCGBmdlHNmGSnhg)|
| Our        | ResNet-50 | F | T | 88.2 | 64.3 | 76.3 | 20.2 | 35.9 | 59.5 | [Baidu](https://pan.baidu.com/s/1zhjXslHz3WkvFWC1hlkV9g) / [Details](https://pan.baidu.com/s/1wiQPjky_PjuGmlQ6Y3AXJA)|
| Our        | ResNet-50 | T | F | 88.3 | 70.3 | 79.4 | 35.3 | 39.3 | 60.2 | [Baidu](https://pan.baidu.com/s/1yRCx49H_a7E_2fweIAJ3Jg) / [Details](https://pan.baidu.com/s/1idfbPPVqQRfWQ_Homqwk1g)|
| Our        | ResNet-50 | T | T | 88.2 | 73.0 | 80.6 | 14.7 | 36.6 | 62.1 | [Baidu](https://pan.baidu.com/s/1UeIJAUm6loGXRN9ZqcKx9g) / [Details](https://pan.baidu.com/s/1dFrYTsOGQwVhWJ4LWR7_Nw)|
