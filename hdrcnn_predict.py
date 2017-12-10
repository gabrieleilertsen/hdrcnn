"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2017, Gabriel Eilertsen.
 " All rights reserved.
 " 
 " Redistribution and use in source and binary forms, with or without 
 " modification, are permitted provided that the following conditions are met:
 " 
 " 1. Redistributions of source code must retain the above copyright notice, 
 "    this list of conditions and the following disclaimer.
 " 
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 " 
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software 
 "    without specific prior written permission.
 " 
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: TensorFlow prediction script, for reconstructing HDR images
                from single expousure LDR images.
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: Aug 2017
"""

import os, sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import network, img_io

eps = 1e-5

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()


# Settings, using TensorFlow arguments
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("width", "1024", "Reconstruction image width")
tf.flags.DEFINE_integer("height", "768", "Reconstruction image height")
tf.flags.DEFINE_string("im_dir", "data", "Path to image directory or an individual image")
tf.flags.DEFINE_string("out_dir", "out", "Path to output directory")
tf.flags.DEFINE_string("params", "hdrcnn_params.npz", "Path to trained CNN weights")
tf.flags.DEFINE_float("scaling", "1.0", "Pre-scaling, which is followed by clipping, in order to remove compression artifacts close to highlights")
tf.flags.DEFINE_float("gamma", "1.0", "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")

# Round to be multiple of 32, so that autoencoder pooling+upsampling
# yields same size as input image
sx = int(np.maximum(32, np.round(FLAGS.width/32.0)*32))
sy = int(np.maximum(32, np.round(FLAGS.height/32.0)*32))
if sx != FLAGS.width or sy != FLAGS.height:
    print_("Warning: ", 'w', True)
    print_("prediction size has been changed from %dx%d pixels to %dx%d\n"%(FLAGS.width, FLAGS.height, sx, sy), 'w')
    print_("         pixels, to comply with autoencoder pooling and up-sampling.\n\n", 'w')

# Info
print_("\n\n\t-------------------------------------------------------------------\n", 'm')
print_("\t  HDR image reconstruction from a single exposure using deep CNNs\n\n", 'm')
print_("\t  Prediction settings\n", 'm')
print_("\t  -------------------\n", 'm')
print_("\t  Input image directory/file:     %s\n" % FLAGS.im_dir, 'm')
print_("\t  Output directory:               %s\n" % FLAGS.out_dir, 'm')
print_("\t  CNN weights:                    %s\n" % FLAGS.params, 'm')
print_("\t  Prediction resolution:          %dx%d pixels\n" % (sx, sy), 'm')
if FLAGS.scaling > 1.0:
    print_("\t  Pre-scaling:                    %0.4f\n" % FLAGS.scaling, 'm')
if FLAGS.gamma > 1.0 + eps or FLAGS.gamma < 1.0 - eps:
    print_("\t  Gamma:                          %0.4f\n" % FLAGS.gamma, 'm')
print_("\t-------------------------------------------------------------------\n\n\n", 'm')

# Single frame
frames = [FLAGS.im_dir]

# If directory is supplied, get names of all files in the path
if os.path.isdir(FLAGS.im_dir):
    frames = [os.path.join(FLAGS.im_dir, name)
              for name in sorted(os.listdir(FLAGS.im_dir))
              if os.path.isfile(os.path.join(FLAGS.im_dir, name))]

# Placeholder for image input
x = tf.placeholder(tf.float32, shape=[1, sy, sx, 3])

# HDR reconstruction autoencoder model
print_("Network setup:\n")
net = network.model(x)

# The CNN prediction (this also includes blending with input image x)
y = network.get_final(net, x)

# TensorFlow session for running inference
sess = tf.InteractiveSession()

# Load trained CNN weights
print_("\nLoading trained parameters from '%s'..."%FLAGS.params)
load_params = tl.files.load_npz(name=FLAGS.params)
tl.files.assign_params(sess, load_params, net)
print_("\tdone\n")

if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

print_("\nStarting prediction...\n\n")
k = 0
for i in range(len(frames)):
    print("Frame %d: '%s'"%(i,frames[i]))

    try:
        # Read frame
        print_("\tReading...")
        x_buffer = img_io.readLDR(frames[i], (sy,sx), True, FLAGS.scaling)
        print_("\tdone")

        print_("\t(Saturation: %0.2f%%)\n" % (100.0*(x_buffer>=1).sum()/x_buffer.size), 'm')

        # Run prediction.
        # The gamma value is used to allow for boosting/reducing the intensity of
        # the reconstructed highlights. If y = f(x) is the reconstruction, the gamma
        # g alters this according to y = f(x^(1/g))^g
        print_("\tInference...")
        feed_dict = {x: np.power(np.maximum(x_buffer, 0.0), 1.0/FLAGS.gamma)}
        y_predict = sess.run([y], feed_dict=feed_dict)
        y_predict = np.power(np.maximum(y_predict, 0.0), FLAGS.gamma)
        print_("\tdone\n")

        # Gamma corrected output
        y_gamma = np.power(np.maximum(y_predict, 0.0), 0.5)

        # Write to disc
        print_("\tWriting...")
        k += 1;
        img_io.writeLDR(x_buffer, '%s/%06d_in.png' % (FLAGS.out_dir, k), -3)
        img_io.writeLDR(y_gamma, '%s/%06d_out.png' % (FLAGS.out_dir, k), -3)
        img_io.writeEXR(y_predict, '%s/%06d_out.exr' % (FLAGS.out_dir, k))
        print_("\tdone\n")

    except img_io.IOException as e:
        print_("\n\t\tWarning! ", 'w', True)
        print_("%s\n"%e, 'w')
    except Exception as e:    
        print_("\n\t\tError: ", 'e', True)
        print_("%s\n"%e, 'e')

print_("Done!\n")

sess.close()

