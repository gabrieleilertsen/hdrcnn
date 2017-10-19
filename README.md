# **Deep learning HDR image reconstruction**

![image](http://hdrv.org/hdrcnn/img/teaser.jpg)

## General
This repository provides code for running inference with the autoencoder convolutional neural network (CNN) described in our [Siggraph Asia paper](http://hdrv.org/hdrcnn/). Please read the information below in order to make proper use of the method. If you use the code for your research work, please consider citing the paper according to:

```
@article{EKDMU17,
  author       = "Eilertsen, Gabriel and Kronander, Joel, and Denes, Gyorgy and Mantiuk, Rafał and Unger, Jonas",
  title        = "HDR image reconstruction from a single exposure using deep CNNs",
  journal      = "ACM Transactions on Graphics (TOG)",
  number       = "6",
  volume       = "36",
  articleno    = "178",
  year         = "2017"
}
```

The CNN is trained to reconstruct image regions where information has been
lost due to sensor saturation, such as highlights and bright image features. 
This means that a standard 8-bit single exposed image can be fed to the network,
which then reconstructs the missing information in order to create a high
dynamic range (HDR) image. Please see the [project webpage](http://hdrv.org/hdrcnn/)
for more information on the method.

## Code specification
The model and prediction scripts are written in Python using the following packages:

* [TensorFlow](https://www.tensorflow.org/) for model specification and prediction.
* [TensorLayer](https://tensorlayer.readthedocs.io/en/latest/) for simplified TensorFlow layer construction.
* [OpenEXR](http://www.openexr.com/) in order to write reconstructed HDR images to disc.
* [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org/) for image handling etc.

All dependencies are available through the Python **pip** package manager (replace 
`tensorflow` with `tensorflow-gpu` for GPU support):

```
$ pip install numpy scipy tensorflow tensorlayer OpenEXR
```

## Usage
1. Trained CNN weights to be used for the inference, can be found [here](http://hdrv.org/hdrcnn/material/hdrcnn_params.npz).
2. Run `python hdrcnn_predict.py -h` to display available input options.
3. Below follows an example to demonstrate how to make an HDR reconstruction.

#### Example
There are a few test images provided, that can be used for reconstruction as follows:

```
$ python hdrcnn_predict.py --params hdrcnn_params.npz --im_dir data --width 1024 --height 768
```

Prediction can also be made on individual frames:

```
$ python hdrcnn_predict.py --params hdrcnn_params.npz --im_dir data/img_001.png --width 1024 --height 768
``` 

#### Compression artifacts
There is a limit to how much compression artifacts that can be present in the input images. Compression can often cause small (and often invisible) blocking artifacts close to highlights, which impairs the HDR reconstruction. Preferably, the input images should contain no or little compression (PNG or JPEG with highest quality setting).

**[UPDATE, 2017-10-19]:** There are now parameters available that have been trained with images that include JPEG compression artifacts. These can be downloaded [here](http://hdrv.org/hdrcnn/material/hdrcnn_params_compr.npz). If the images for reconstruction contain compression artifacts, these parameters makes for a substantial improvement in reconstruction quality as compared to the previous parameters. However, if the input images contain no compression artifacts we recommend to use the [original parameters](http://hdrv.org/hdrcnn/material/hdrcnn_params.npz) as these allow for a slight advantage in terms of reconstructed details.

## License

Copyright (c) 2017, Gabriel Eilertsen.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
