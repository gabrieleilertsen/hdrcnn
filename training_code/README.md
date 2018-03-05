# **HDR-CNN training code**
Here you can find the resources for performing training of the HDR reconstruction CNN. There is one training script, and a pre-processing/data augmentation application. This application acts as a virtual camera, capturing a set of crops from an HDR scene, and simulating different camera parameters. The application is called from within the training script.

## Setup
While the prediction code in the parent folder is straightforward and with minimal dependencies, pretty much running out-of-the-box, the training pipeline may require more preparation in terms of data, dependencies and parameters. Furthermore, there is not proper exception handling in all places. The code has only been tested under linux (Fedora 25), so for other platforms some tweaking is probably required.


### Dependencies
The dependencies for the training script are pretty much the same as for the inference code. However, the preprocessing application needs to be compiled with OpenCV. 

For the virtual camera pre-processing, OpenCV is used both to read HDR images, and to output LDR JPEG images. Thus, OpenCV needs to be compiled with support for reading the training images, e.g. using OpenEXR and/or Radiance RGBE, and with support for JPEG.

The training script uses the same model and image I/O as the prediction script. Thus, it requires this code to be in the parental folder, or any other path known to Python.

### Installation
In order to run a training session, compile the virtual camera application before executing the training script:

```
$ cd virtualcamera/
$ gcc -Wall -lm -lstdc++ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs virtualcamera.cpp -o virtualcamera
$ cd ..
$ python hdrcnn_training.py
```

## Training procedure
See `hdrcnn_training.py` for a full list of possible input arguments, or display these by running `python hdrcnn_training.py -h`. Make sure that the different paths are specified properly, so that training data can be found etc. The `--raw_dir` specifies the path to the input training images. This folder will be traveresed recursively by the virtual camera application, loading all the images for processing and augmentation.

For training of the final weights provided with the HDR-CNN prediction script, a two-stage training procedure was performed. This is briefly described next.

### 1. Pre-training
The pre-training stage is performed on a simulated database of HDR images. The data augmentation application will create this database from LDR images. For the pre-training in the [Siggraph Asia paper](http://hdrv.org/hdrcnn/) LDR images with no saturation were choosen from the Places dataset, see paper for details. The option `--linearize` applies an inverse camera curve before data augmentation, in order to approximate linear HDR pixel values. The option `--sub_im_min_jpg` is set to store LDR images with maximum JPEG quality, assuming that there already is lossy compression applied to the input images. Finally, the encoder weights are initialized using VGG16 weights trained for classification. Weights converted to TensorFlow can be found [here](http://hdrv.org/hdrcnn/material/vgg16_places365_weights.npy).

Given the afore-mentioned considerations, an example pre-training command may look something like this:

```
$ python3 -u hdrcnn_train.py \
    --raw_dir "PATH_TO_LDR_DATABASE" \
    --vgg_path "PATH_TO_VGG16_WEIGHTS" \
    --sx 224 --sy 224 \
    --preprocess 1 \
    --batch_size 16 \
    --learning_rate 0.00002 \
    --sub_im 0 \
    --sub_im_sc1 0.5 --sub_im_sc2 0.8 \
    --sub_im_clip1 0.7 --sub_im_clip2 0.9 \
    --sub_im_min_jpg 100 \
    --linearize 1 
```

### 2. Fine-tuning
After performing the pre-training, the optimized CNN weights are used to initilize the network, followed by training on the main HDR dataset. The dataset is gathered from online resources, as listed in the supplementary document of the [Siggraph Asia paper](http://hdrv.org/hdrcnn/). Depending on GPU memory, the batch size may need to be reduced as compared to the pre-training, since default image size is 320x320 pixels.

The training can be executed as follows:
```
$ python3 -u hdrcnn_train.py \
    --raw_dir    "PATH_TO_HDR_DATABASE" \
    --parameters "PATH_TO_WEIGHTS_FROM_PRETRAINING" \
    --load_params 1 \
    --preprocess 1 \
    --batch_size 8
```




