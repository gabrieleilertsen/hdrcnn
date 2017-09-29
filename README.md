# **Deep learning HDR image reconstruction**

## General
This repository provides code for running inference with the autoencoder convolutional neural
network (CNN) described in our [Siggraph Asia paper](http://vcl.itn.liu.se/publications/2017/EKDMU17/). If you use the code for your research work, please consider citing the paper
according to:

```
@Article{EKDMU15,
  author       = "Eilertsen, Gabriel and Kronander, Joel, and Denes, Gyorgy and Mantiuk, Rafał and Unger, Jonas",
  title        = "HDR image reconstruction from a single exposures using deep CNNs",
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
dynamic range (HDR) image.

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
1. Trained CNN weights to be used for the inference, can be found here:
[trained_params.npz](trained_params.npz)
2. Run `python hdrcnn_predict.py -h` to display available input options.
3. Below follows an example to demonstrate how to make an HDR reconstruction.

#### Example
There are a few test images provided, that can be used for reconstruction as follows:

```
$ python hdrcnn_predict.py --params trained_params.npz --im_dir data --width 1024 --height 768
```

Prediction can also be made on individual frames:

```
$ python hdrcnn_predict.py --params trained_params.npz --im_dir data/img_001.png --width 1024 --height 768
``` 

#### Compression artifacts
There is a limit to how much compression artifacts that can be present in the input images. Compression can often cause small blocking artifacts close to highlights, which impairs the HDR reconstruction. Preferably, the input images should contain no or little compression (PNG or JPEG with highest quality setting). If the images for reconstruction are not available without compression artifacts, there is a simple trick that be used to alleviate the problem: scale the images somewhat followed by clipping,

```
I = min(1, sI),
```

where `I` is the input image and `s` is the scaling (`s > 1`). This modification will remove some of the compression artifacts close to saturated image regions, which improves reconstruction performance at the cost of some loss of details. There is also an input argument `--scaling`, which specifies the scaling `s` in the equation above, that can be used for doing this when reading images for reconstruction:

```
$ python hdrcnn_predict.py --im_dir some_image.jpg --scaling 1.2
```

## License

Copyright (c) 2017, Gabriel Eilertsen.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
