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
 " Description: Training script for the HDR-CNN
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: February 2018
"""

import time, math, os, sys, random
import tensorflow as tf
import tensorlayer as tl
import threading
import numpy as np
import scipy.stats as st

sys.path.insert(0, "../")
import network, img_io

eps = 1.0/255.0


#=== Settings =================================================================

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("sx",               "320",   "Image width")
tf.flags.DEFINE_integer("sy",               "320",   "Image height")
tf.flags.DEFINE_integer("num_threads",      "4",     "Number of threads for multi-threaded loading of data")
tf.flags.DEFINE_integer("print_batch_freq", "5000",  "Frequency for printing stats and saving images/parameters")
tf.flags.DEFINE_integer("print_batches",    "5",     "Number of batches to output images for at each [print_batch_freq] step")
tf.flags.DEFINE_bool("print_im",            "true",  "If LDR sample images should be printed at each [print_batch_freq] step")
tf.flags.DEFINE_bool("print_hdr",           "false", "If HDR reconstructions should be printed at each [print_batch_freq] step")

# Paths
tf.flags.DEFINE_string("raw_dir",           "input_data", "Path to unprocessed dataset")
tf.flags.DEFINE_string("data_dir",          "training_data", "Path to processed dataset. This data will be created if the flag [preprocess] is set")
tf.flags.DEFINE_string("output_dir",        "training_output", "Path to output directory, for weights and intermediate results")
tf.flags.DEFINE_string("vgg_path",          "weights/vgg16_places365_weights.npy", "Path to VGG16 pre-trained weigths, for encoder convolution layers")
tf.flags.DEFINE_string("parameters",        "weights/model_trained.npz", "Path to trained params for complete network")
tf.flags.DEFINE_bool("load_params",         "false", "Load the parameters from the [parameters] path, otherwise the parameters from [vgg_path] will be used")

# Data augmentation parameters
tf.flags.DEFINE_bool("preprocess",          "false", "Pre-process HDR input data, to create augmented dataset for training")
tf.flags.DEFINE_integer("sub_im",           "10",    "Number of subimages to pick in a 1 MP pixel image")
tf.flags.DEFINE_integer("sub_im_linearize", "0",     "Linearize input images")
tf.flags.DEFINE_float("sub_im_sc1",         "0.2",   "Min size of crop, in fraction of input image")
tf.flags.DEFINE_float("sub_im_sc2",         "0.6",   "Max size of crop, in fraction of input image")
tf.flags.DEFINE_float("sub_im_clip1",       "0.85",  "Min saturation limit, i.e. min fraction of non-saturated pixels")
tf.flags.DEFINE_float("sub_im_clip2",       "0.95",  "Max saturation limit, i.e. max fraction of non-saturated pixels")
tf.flags.DEFINE_float("sub_im_noise1",      "0.0",   "Min noise std")
tf.flags.DEFINE_float("sub_im_noise2",      "0.01",  "Max noise std")
tf.flags.DEFINE_float("sub_im_hue_mean",    "0.0",   "Mean hue")
tf.flags.DEFINE_float("sub_im_hue_std",     "7.0",   "Std of hue")
tf.flags.DEFINE_float("sub_im_sat_mean",    "0.0",   "Mean saturation")
tf.flags.DEFINE_float("sub_im_sat_std",     "0.1",   "Std of saturation")
tf.flags.DEFINE_float("sub_im_sigmn_mean",  "0.9",   "Mean sigmoid exponent")
tf.flags.DEFINE_float("sub_im_sigmn_std",   "0.1",   "Std of sigmoid exponent")
tf.flags.DEFINE_float("sub_im_sigma_mean",  "0.6",   "Mean sigmoid offset")
tf.flags.DEFINE_float("sub_im_sigma_std",   "0.1",   "Std of sigmoid offset")
tf.flags.DEFINE_integer("sub_im_min_jpg",   "30",    "Minimum quality level of generated LDR images")

# Learning parameters
tf.flags.DEFINE_float("num_epochs",         "100.0",   "Number of training epochs")
tf.flags.DEFINE_float("start_step",         "0.0",     "Step to start from")
tf.flags.DEFINE_float("learning_rate",      "0.00005", "Starting learning rate for Adam optimizer")
tf.flags.DEFINE_integer("batch_size",       "4",       "Batch size for training")
tf.flags.DEFINE_bool("sep_loss",            "true",    "Use illumination + reflectance loss")
tf.flags.DEFINE_float("lambda_ir",          "0.5",     "Reflectance weight for the ill+refl loss")
tf.flags.DEFINE_bool("rand_data",           "true",    "Random shuffling of training data")
tf.flags.DEFINE_float("train_size",         "0.99",    "Fraction of data to use for training, the rest is validation data")
tf.flags.DEFINE_integer("buffer_size",      "256",     "Size of load queue when reading training data")


#==============================================================================

sx =  FLAGS.sx
sy =  FLAGS.sy
data_dir_bin = os.path.join(FLAGS.data_dir, "bin")
data_dir_jpg = os.path.join(FLAGS.data_dir, "jpg")
log_dir = os.path.join(FLAGS.output_dir, "logs")
im_dir = os.path.join(FLAGS.output_dir, "im")


#=== Pre-processing/data augmentation =========================================

# Process training data
if (FLAGS.preprocess):
    cmd = "./virtualcamera/virtualcamera -linearize %d -imsize %d %d 3 -input_path %s -output_path %s \
                                   -subimages %d -cropscale %f %f -clip %f %f -noise %f %f \
                                   -hue %f %f -sat %f %f -sigmoid_n %f %f -sigmoid_a %f %f \
                                   -jpeg_quality %d" % \
                                   (FLAGS.sub_im_linearize, sy, sx, FLAGS.raw_dir, FLAGS.data_dir, FLAGS.sub_im,
                                    FLAGS.sub_im_sc1, FLAGS.sub_im_sc2,
                                    FLAGS.sub_im_clip1, FLAGS.sub_im_clip2,
                                    FLAGS.sub_im_noise1, FLAGS.sub_im_noise2,
                                    FLAGS.sub_im_hue_mean, FLAGS.sub_im_hue_std,
                                    FLAGS.sub_im_sat_mean, FLAGS.sub_im_sat_std, 
                                    FLAGS.sub_im_sigmn_mean, FLAGS.sub_im_sigmn_std, 
                                    FLAGS.sub_im_sigma_mean, FLAGS.sub_im_sigma_std,
                                    FLAGS.sub_im_min_jpg);
    print("\nRunning processing of training data")
    print("cmd = '%s'\n\n"%cmd)

    # Remove old data, and run new data generation
    os.system("rm -rf %s"%FLAGS.data_dir)
    os.makedirs(data_dir_bin)
    os.makedirs(data_dir_jpg)
    os.system(cmd)
    print("\n")

# Create output directories
tl.files.exists_or_mkdir(log_dir)
tl.files.exists_or_mkdir(im_dir)


#=== Localize training data ===================================================

# Get names of all images in the training path
frames = [name for name in sorted(os.listdir(data_dir_bin)) if os.path.isfile(os.path.join(data_dir_bin, name))]

# Randomize the images
if FLAGS.rand_data:
    random.shuffle(frames)

# Split data into training/validation sets
splitPos = len(frames) - math.floor(max(FLAGS.batch_size, min((1-FLAGS.train_size)*len(frames), 1000)))
frames_train, frames_valid = np.split(frames, [splitPos])

# Number of steps per epoch depends on the number of training images
training_samples = len(frames_train)
validation_samples = len(frames_valid)
steps_per_epoch = training_samples/FLAGS.batch_size

print("\n\nData to be used:")
print("\t%d training images" % training_samples)
print("\t%d validation images\n" % validation_samples)


#=== Load validation data =====================================================

# Load all validation images into memory
print("Loading validation data...")
x_valid, y_valid = [], []
for i in range(len(frames_valid)):
    if i % 10 == 0:
        print("\tframe %d of %d" % (i, len(frames_valid)))
    
    succ, xv, yv = img_io.load_training_pair(os.path.join(data_dir_bin, frames_valid[i]), os.path.join(data_dir_jpg, frames_valid[i].replace(".bin", ".jpg")))
    if not succ:
        continue
    xv = xv[np.newaxis,:,:,:]
    yv = yv[np.newaxis,:,:,:]

    if i == 0:
        x_valid, y_valid = xv, yv
    else:
        x_valid = np.concatenate((x_valid, xv), axis=0)
        y_valid = np.concatenate((y_valid, yv), axis=0)
print("...done!\n\n")

del frames


#=== Setup data queues ========================================================

# For single-threaded queueing of frame names
input_frame = tf.placeholder(tf.string)
q_frames = tf.FIFOQueue(FLAGS.buffer_size, [tf.string])
enqueue_op_frames = q_frames.enqueue([input_frame])
dequeue_op_frames = q_frames.dequeue()

# For multi-threaded queueing of training images
input_data = tf.placeholder(tf.float32, shape=[sy, sx, 3])
input_target = tf.placeholder(tf.float32, shape=[sy, sx, 3])
q_train = tf.FIFOQueue(FLAGS.buffer_size, [tf.float32, tf.float32], shapes=[[sy,sx,3], [sy,sx,3]])
enqueue_op_train = q_train.enqueue([input_target, input_data])
y_, x = q_train.dequeue_many(FLAGS.batch_size)


#=== Network ==================================================================

# Setup the network
print("Network setup:\n")
net, vgg16_conv_layers = network.model(x, FLAGS.batch_size, True)

y = net.outputs
train_params = net.all_params

# The TensorFlow session to be used
sess = tf.InteractiveSession()


#=== Loss function formulation ================================================

# For masked loss, only using information near saturated image regions
thr = 0.05 # Threshold for blending
msk = tf.reduce_max(y_, reduction_indices=[3])
msk = tf.minimum(1.0, tf.maximum(0.0, msk-1.0+thr)/thr)
msk = tf.reshape(msk, [-1, sy, sx, 1])
msk = tf.tile(msk, [1,1,1,3])

# Loss separated into illumination and reflectance terms
if FLAGS.sep_loss:
    y_log_ = tf.log(y_+eps)
    x_log = tf.log(tf.pow(x, 2.0)+eps)

    # Luminance
    lum_kernel = np.zeros((1, 1, 3, 1))
    lum_kernel[:, :, 0, 0] = 0.213
    lum_kernel[:, :, 1, 0] = 0.715
    lum_kernel[:, :, 2, 0] = 0.072
    y_lum_lin_ = tf.nn.conv2d(y_, lum_kernel, [1, 1, 1, 1], padding='SAME')
    y_lum_lin = tf.nn.conv2d(tf.exp(y)-eps, lum_kernel, [1, 1, 1, 1], padding='SAME')
    x_lum_lin = tf.nn.conv2d(x, lum_kernel, [1, 1, 1, 1], padding='SAME')

    # Log luminance
    y_lum_ = tf.log(y_lum_lin_ + eps)
    y_lum = tf.log(y_lum_lin + eps)
    x_lum = tf.log(x_lum_lin + eps)

    # Gaussian kernel
    nsig = 2
    filter_size = 13
    interval = (2*nsig+1.)/(filter_size)
    ll = np.linspace(-nsig-interval/2., nsig+interval/2., filter_size+1)
    kern1d = np.diff(st.norm.cdf(ll))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    # Illumination, approximated by means of Gaussian filtering
    weights_g = np.zeros((filter_size, filter_size, 1, 1))
    weights_g[:, :, 0, 0] = kernel
    y_ill_ = tf.nn.conv2d(y_lum_, weights_g, [1, 1, 1, 1], padding='SAME')
    y_ill = tf.nn.conv2d(y_lum, weights_g, [1, 1, 1, 1], padding='SAME')
    x_ill = tf.nn.conv2d(x_lum, weights_g, [1, 1, 1, 1], padding='SAME')

    # Reflectance
    y_refl_ = y_log_ - tf.tile(y_ill_, [1,1,1,3])
    y_refl = y - tf.tile(y_ill, [1,1,1,3])
    x_refl = x - tf.tile(x_ill, [1,1,1,3])

    cost =              tf.reduce_mean( ( FLAGS.lambda_ir*tf.square( tf.subtract(y_ill, y_ill_) ) + (1.0-FLAGS.lambda_ir)*tf.square( tf.subtract(y_refl, y_refl_) ) )*msk )
    cost_input_output = tf.reduce_mean( ( FLAGS.lambda_ir*tf.square( tf.subtract(x_ill, y_ill_) ) + (1.0-FLAGS.lambda_ir)*tf.square( tf.subtract(x_refl, y_refl_) ) )*msk )
else:
    cost =              tf.reduce_mean( tf.square( tf.subtract(y, tf.log(y_+eps) )*msk ) )
    cost_input_output = tf.reduce_mean( tf.square( tf.subtract(tf.log(y_+eps), tf.log(tf.pow(x, 2.0)+eps) )*msk ) );

# Optimizer
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = FLAGS.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           int(steps_per_epoch), 0.99, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-8, use_locking=False).minimize(cost, global_step=global_step, var_list = train_params)


#=== Data enqueueing functions ================================================

# For enqueueing of frame names
def enqueue_frames(enqueue_op, coord, frames):
    
    num_frames = len(frames)
    i, k = 0, 0

    try:
        while not coord.should_stop():
            if k >= training_samples*FLAGS.num_epochs:
                    sess.run(q_frames.close())
                    break

            if i == num_frames:
                i = 0
                if FLAGS.rand_data:
                    random.shuffle(frames)

            fname = frames[i];

            i += 1
            k += 1
            sess.run(enqueue_op, feed_dict={input_frame: fname})
    except tf.errors.OutOfRangeError:
        pass
    except Exception as e:
        coord.request_stop(e)

# For multi-threaded reading and enqueueing of frames
def load_and_enqueue(enqueue_op, coord):
    try:
        while not coord.should_stop():
            fname = sess.run(dequeue_op_frames).decode("utf-8")

            # Load pairs of HDR/LDR images
            succ, input_data_r, input_target_r = img_io.load_training_pair(os.path.join(data_dir_bin, fname), os.path.join(data_dir_jpg, fname.replace(".bin", ".jpg")))
            if not succ:
                continue
            sess.run(enqueue_op, feed_dict={input_data: input_data_r, input_target: input_target_r})
    except Exception as e:
        try:
            sess.run(q_train.close())
        except Exception as e:
            pass


#=== Error and output function ================================================

# For calculation of loss and output of intermediate validations images to disc
def calc_loss_and_print(x_data, y_data, print_dir, step, N):
    val_loss, orig_loss, n_batch = 0, 0, 0
    for b in range(int(x_data.shape[0]/FLAGS.batch_size)):
        x_batch = x_data[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
        y_batch = y_data[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
        feed_dict = {x: x_batch, y_: y_batch}
        err1, err2, y_predict, y_gt, M = sess.run([cost, cost_input_output, y, y_, msk], feed_dict=feed_dict)


        val_loss += err1; orig_loss += err2; n_batch += 1
        batch_dir = print_dir

        if x_data.shape[0] > x_batch.shape[0]:
            batch_dir = '%s/batch_%03d' % (print_dir, n_batch)

        if n_batch <= N or N < 0:
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)
            for i in range(0, x_batch.shape[0]):
                yy_p = np.squeeze(y_predict[i])
                xx = np.squeeze(x_batch[i])
                yy = np.squeeze(y_gt[i])
                mm = np.squeeze(M[i])

                # Apply inverse camera curve
                x_lin = np.power(np.divide(0.6*xx, np.maximum(1.6-xx, 1e-10) ), 1.0/0.9)

                # Transform log predictions to linear domain
                yy_p = np.exp(yy_p)-eps

                # Masking
                y_final = (1-mm)*x_lin + mm*yy_p

                # Gamma correction
                yy_p = np.power(np.maximum(yy_p, 0.0), 0.5)
                y_final = np.power(np.maximum(y_final, 0.0), 0.5)
                yy = np.power(np.maximum(yy, 0.0), 0.5)
                xx = np.power(np.maximum(x_lin, 0.0), 0.5)

                # Print LDR samples
                if FLAGS.print_im:
                    img_io.writeLDR(xx, "%s/%06d_%03d_in.png" % (batch_dir, step, i+1), -3)
                    img_io.writeLDR(yy, "%s/%06d_%03d_gt.png" % (batch_dir, step, i+1), -3)
                    img_io.writeLDR(y_final, "%s/%06d_%03d_out.png" % (batch_dir, step, i+1), -3)
                
                # Print HDR samples
                if FLAGS.print_hdr:
                    img_io.writeEXR(xx, "%s/%06d_%03d_in.exr" % (batch_dir, step, i+1))
                    img_io.writeEXR(yy, "%s/%06d_%03d_gt.exr" % (batch_dir, step, i+1))
                    img_io.writeEXR(y_final, "%s/%06d_%03d_out.exr" % (batch_dir, step, i+1))

    return (val_loss/n_batch, orig_loss/n_batch)


#=== Setup threads and load parameters ========================================

# Summary for Tensorboard
tf.summary.scalar("learning_rate", learning_rate)
summaries = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(log_dir, sess.graph)

sess.run(tf.global_variables_initializer())

# Threads and thread coordinator
coord = tf.train.Coordinator()
thread1 = threading.Thread(target=enqueue_frames, args=[enqueue_op_frames, coord, frames_train])
thread2 = [threading.Thread(target=load_and_enqueue, args=[enqueue_op_train, coord]) for i in range(FLAGS.num_threads)]
thread1.start()
for t in thread2:
    t.start()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Loading model weights
if(FLAGS.load_params):
    # Load model weights
    print("\n\nLoading trained parameters from '%s'..." % FLAGS.parameters)
    load_params = tl.files.load_npz(name=FLAGS.parameters)
    tl.files.assign_params(sess, load_params, net)
    print("...done!\n")
else:
    # Load pretrained VGG16 weights for encoder
    print("\n\nLoading parameters for VGG16 convolutional layers, from '%s'..." % FLAGS.vgg_path)
    network.load_vgg_weights(vgg16_conv_layers, FLAGS.vgg_path, sess)
    print("...done!\n")


#=== Run training loop ========================================================

print("\nStarting training...\n")

step = FLAGS.start_step
train_loss = 0.0
start_time = time.time()
start_time_tot = time.time()


# The training loop
try:
    while not coord.should_stop():
        step += 1

        _, err_t = sess.run([train_op,cost])

        train_loss += err_t

        # Statistics on intermediate progress
        v = int(max(1.0,FLAGS.print_batch_freq/5.0))
        if (int(step) % v)  == 0:
            val_loss, n_batch = 0, 0

            # Validation loss
            for b in range(int(x_valid.shape[0]/FLAGS.batch_size)):
                x_batch = x_valid[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
                y_batch = y_valid[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
                feed_dict = {x: x_batch, y_: y_batch}
                err = sess.run(cost, feed_dict=feed_dict)
                val_loss += err; n_batch += 1

            # Training and validation loss for Tensorboard
            train_summary = tf.Summary()
            valid_summary = tf.Summary()
            valid_summary.value.add(tag='validation_loss',simple_value=val_loss/n_batch)
            file_writer.add_summary(valid_summary, step)
            train_summary.value.add(tag='training_loss',simple_value=train_loss/v)
            file_writer.add_summary(train_summary, step)

            # Other statistics for Tensorboard
            summary = sess.run(summaries)
            file_writer.add_summary(summary, step)
            file_writer.flush()
            
            # Intermediate training statistics
            print('  [Step %06d of %06d. Processed %06d of %06d samples. Train loss = %0.6f, valid loss = %0.6f]' % (step, steps_per_epoch*FLAGS.num_epochs, (step % steps_per_epoch)*FLAGS.batch_size, training_samples, train_loss/v, val_loss/n_batch))
            train_loss = 0.0

        # Print statistics, and save weights and some validation images
        if step % FLAGS.print_batch_freq == 0:
            duration = time.time() - start_time
            duration_tot = time.time() - start_time_tot

            print_dir = '%s/step_%06d' % (im_dir, step)
            val_loss, orig_loss = calc_loss_and_print(x_valid, y_valid, print_dir, step, FLAGS.print_batches)

            # Training statistics
            print('\n')
            print('-------------------------------------------')
            print('Currently at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
            print('Valid loss input   = %.5f' % (orig_loss))
            print('Valid loss trained = %.5f' % (val_loss))
            print('Timings:')
            print('       Since last: %.3f sec' % (duration))
            print('         Per step: %.3f sec' % (duration/FLAGS.print_batch_freq))
            print('        Per epoch: %.3f sec' % (duration*steps_per_epoch/FLAGS.print_batch_freq))
            print('')
            print('   Per step (avg): %.3f sec' % (duration_tot/step))
            print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
            print('')
            print('       Total time: %.3f sec' % (duration_tot))
            print('   Exp. time left: %.3f sec' % (duration_tot*steps_per_epoch*FLAGS.num_epochs/step - duration_tot))
            print('-------------------------------------------')

            # Save current weights
            tl.files.save_npz(net.all_params , name=("%s/model_step_%06d.npz"%(log_dir,step)))
            print('\n')

            start_time = time.time()

except tf.errors.OutOfRangeError:
    print('Done!')
except Exception as e:
    print("ERROR: ", e)


#=== Final stats and weights ==================================================

duration = time.time() - start_time
duration_tot = time.time() - start_time_tot

print_dir = '%s/step_%06d' % (im_dir, step)
val_loss, orig_loss = calc_loss_and_print(x_valid, y_valid, print_dir, step, FLAGS.print_batches)

# Final statistics
print('\n')
print('-------------------------------------------')
print('Finished at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
print('Valid loss input   = %.5f' % (orig_loss))
print('Valid loss trained = %.5f' % (val_loss))
print('Timings:')
print('   Per step (avg): %.3f sec' % (duration_tot/step))
print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
print('')
print('       Total time: %.3f sec' % (duration_tot))
print('-------------------------------------------')

# Save final weights
tl.files.save_npz(net.all_params , name=("%s/model_step_%06d.npz"%(log_dir,step)))
print('\n')


#=== Shut down ================================================================

# Stop threads
print("Shutting down threads...")
try:
    coord.request_stop()
except Exception as e:
    print("ERROR: ", e)

# Wait for threads to finish
print("Waiting for threads...")
coord.join(threads)

file_writer.close()
sess.close()