# **Deep learning HDR image reconstruction**

![image](https://computergraphics.on.liu.se/hdrcnn/img/teaser.jpg)

## General
This repository provides code for running inference with the autoencoder convolutional neural network (CNN) described in our [Siggraph Asia paper](https://arxiv.org/abs/1710.07480), as well as training of the network. Please read the information below in order to make proper use of the method. If you use the code for your research work, please consider citing the paper according to:

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

The CNN is trained to reconstruct image regions where information has been lost due to sensor saturation, such as highlights and bright image features. This means that a standard 8-bit single exposed image can be fed to the network, which then reconstructs the missing information in order to create a high dynamic range (HDR) image. Please see the [project webpage](https://computergraphics.on.liu.se/hdrcnn/) for more information on the method.

In what follows are descriptions on how to make HDR reconstructions using the trained network. For training of new weigths information and code is provided in the [training_code](training_code/) folder.

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

You may have to install OpenEXR through the appropriate package manager before pip install (e.g. sudo apt-get install openexr and libopenexr-dev on Ubuntu).

## Usage
1. Trained CNN weights to be used for the inference, can be found [here](https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params.npz).
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
There are parameters available that have been trained with images that include JPEG compression artifacts. These can be downloaded [here](https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params_compr.npz). If the images for reconstruction contain compression artifacts, these parameters makes for a substantial improvement in reconstruction quality as compared to the previous parameters. However, if the input images contain no compression artifacts we recommend to use the [original parameters](https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params.npz) as these allow for a slight advantage in terms of reconstructed details.

#### Video reconstruction
Reconstruction video material frame by frame most often results in flickering artifacts and different local temporal incoherencies. In order to alleviate this problem, we provide parameters [here](https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params_compr_regularized.npz), which have been trained using the regularization method proposed in our CVPR 2019 paper ([paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Eilertsen_Single-Frame_Regularization_for_Temporally_Stable_CNNs_CVPR_2019_paper.html), [project web](https://computergraphics.on.liu.se/temporally_stable_cnns)):

```
@inproceedings{EMU19,
  author       = "Eilertsen, Gabriel and 
                  Mantiuk, Rafa\l and 
                  Unger, Jonas",
  title        = "Single-frame Regularization for Temporally Stable CNNs",
  booktitle    = "The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "June",
  year         = "2019"
}
```
The parameters trained for increased temporal coherence also use JPEG compressed images, so these are possible to use also for video with compression applied. There is some trade-of between reconstruction quality and temporal coherence. If you do not need to reconstruct video material, the [original parameters](https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params_compr_regularized.npz) should be prefered.

For training with the above regularization applied, functionality is avaliable in the [training_code](training_code/).

#### Evaluation
Properly evaluating single-image HDR reconstruction methods is difficult (see, e.g., [here](https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Eilertsen_How_To_Cheat_With_Metrics_in_Single-Image_HDR_Reconstruction_ICCVW_2021_paper.html)). We recommend using the advised evaluation protocols proposed in our SIGGRAPH 2022 paper ([paper](https://dl.acm.org/doi/abs/10.1145/3528233.3530729), [project web](https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/)):

```
@inproceedings{hanji2022sihdr,
  author    = {Hanji, Param and Mantiuk, Rafa{\l} K. and Eilertsen, Gabriel and Hajisharif, Saghi and Unger, Jonas},
  title     = {Comparison of single image HDR reconstruction methods — the caveats of quality assessment},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings (SIGGRAPH '22 Conference Proceedings)},
  year      = {2022},
  doi       = {10.1145/3528233.3530729},
  url       = {https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/},
}
```

#### Controlling the reconstruction
The HDR reconstruction with the CNN is completely automatic, with no parameter calibration needed. However, in some situations it may be beneficial to be able to control how bright the reconstructed pixels will be. To this end, there is a simple trick that can be used to allow for such control.

Given the input image **x**, the CNN prediction **y = f(x)** can be controlled somewhat by altering the input image with an exponential/gamma function, and inverting this after the reconstruction,

&nbsp;&nbsp;&nbsp;&nbsp;**y = f(x<sup>1/g</sup>)<sup>g</sup>**.

Essentially, this modifies the camera curve of the image, so that reconstruction is performed given other camera characteristics. For a value **g > 1**, the intensities of reconstructed bright regions will be boosted, and vice versa for **g < 1**. There is an input option `--gamma` that allows to perform this modification:

```
$ python hdrcnn_predict.py [...] --gamma 1.2
```
In general, a value around *1.1-1.3* may be a good idea, since this prevents underestimation of the brightest pixels, which otherwise is common (e.g. due to limitations in the training data).


## License

Copyright (c) 2017, Gabriel Eilertsen.
All rights reserved.

The code is distributed under a BSD license. See `LICENSE` for information.
