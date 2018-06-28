#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:55:11 2018

@author: durvesh
"""
import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import time
from datetime import timedelta
import cifar10
from cifar10 import img_size, num_channels, num_classes


#data = input_data.read_data_sets('data/MNIST/', one_hot=True)
def download_dataset():
    cifar10.maybe_download_and_extract()
    class_names = cifar10.load_class_names()
    images_train,cls_train,labels_train = cifar10.load_training_data()
    images_test,cls_test,labels_test = cifar10.load_test_data()
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))
    
    return class_names,images_train,cls_train,labels_train,images_test,cls_test,labels_test
class_names,images_train,cls_train,labels_train,images_test,cls_test,labels_test = download_dataset()
def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

#distorted_images = pre_process(images=x, training=True)

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch
def random_batch_test():
    # Number of images in the training-set.
    num_images = len(images_test)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=30,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_test[idx, :, :, :]
    y_batch = labels_test[idx, :]

    return x_batch, y_batch
def fire_module(input, fire_id, channel, s1, e1, e3,):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e3: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor
    """

    fire_weights = {'conv_s_1': tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, channel, s1])),
                    'conv_e_1': tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, s1, e1])),
                    'conv_e_3': tf.Variable(tf.contrib.layers.xavier_initializer()([3, 3, s1, e3]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.contrib.layers.xavier_initializer()([s1])),
                   'conv_e_1': tf.Variable(tf.contrib.layers.xavier_initializer()([e1])),
                   'conv_e_3': tf.Variable(tf.contrib.layers.xavier_initializer()([e3]))}

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.elu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        result = tf.nn.relu(result)
        return result


def squeeze_netv_1_1(inputs, classes):
    """
    SqueezeNet model written in tensorflow. It provides AlexNet level accuracy with 50x fewer parameters
    and smaller model size.
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor
    """

    weights = {'conv1_1': tf.Variable(tf.contrib.layers.xavier_initializer()([3, 3, 3, 64])),
                
               'conv1_2': tf.Variable(tf.contrib.layers.xavier_initializer()([3, 3, 64, 64])),
               
               'conv5_2': tf.Variable(tf.contrib.layers.xavier_initializer()([3,3,64,32])),
               
               'conv5_3': tf.Variable(tf.contrib.layers.xavier_initializer()([3,3,32,32])),
               
               'conv5_4': tf.Variable(tf.contrib.layers.xavier_initializer()([3,3,32,32])),
               
               'conv5_5': tf.Variable(tf.contrib.layers.xavier_initializer()([3,3,32,10]))
               
               }

    biases = {'conv1_1': tf.Variable(tf.contrib.layers.xavier_initializer()([64])),
                                     
              'conv1_2': tf.Variable(tf.contrib.layers.xavier_initializer()([64])),
              
              'conv5_2': tf.Variable(tf.contrib.layers.xavier_initializer()([32])),
              
              'conv5_3': tf.Variable(tf.contrib.layers.xavier_initializer()([32])),
              
              'conv5_4': tf.Variable(tf.contrib.layers.xavier_initializer()([32])),
               
              'conv5_5': tf.Variable(tf.contrib.layers.xavier_initializer()([10]))
              }
             

    # first layer of convs
    
    output = tf.nn.conv2d(inputs, weights['conv1_1'], strides=[1,1,1,1], padding='SAME', name='conv1_1')
    output = tf.nn.bias_add(output, biases['conv1_1'])
    output = tf.nn.relu(output)
   # output = tf.nn.dropout(output, keep_prob=0.5, name='dropout11')
    output = tf.nn.conv2d(output,weights['conv1_2'],strides=[1,1,1,1], padding='SAME', name='conv1_2')
    output = tf.nn.bias_add(output,biases['conv1_2'])
    output = tf.nn.relu(output)
   # output = tf.nn.dropout(output, keep_prob=0.5, name='dropout12')
    output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name='maxpool1')
    
    # second layer of fire convs
    
    output = fire_module(output, s1=32, e1=64, e3=64, channel=64, fire_id='fire2')
    output = fire_module(output, s1=32, e1=64, e3=64, channel=128, fire_id='fire3')
    output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name='maxpool2')
   # output = tf.nn.dropout(output, keep_prob=0.5, name='dropout23')
    # third layer of fire convs
    
    output = fire_module(output,s1=64, e1=128,e3=128, channel=128,fire_id='fire4')
    output = fire_module(output,s1=64, e1=128,e3=128, channel=256,fire_id='fire5')
    output = fire_module(output,s1=64, e1=128,e3=128, channel=256,fire_id='fire6')
    output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME', name='maxpool3')
    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout1')
    # fourth layer of fire convs
    
    output = fire_module(output,s1=128, e1=256,e3=256, channel=256,fire_id='fire7')
    output = fire_module(output,s1=128, e1=128,e3=128, channel=512,fire_id='fire8')
    output = fire_module(output,s1=64, e1=64,e3=64, channel=256,fire_id='fire9')
    output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME', name='maxpool4')
    
    output = fire_module(output,s1=32, e1=32,e3=32, channel=128,fire_id='fire10')
    output = tf.nn.conv2d(output,weights['conv5_2'], strides=[1,1,1,1], padding='SAME', name='conv5_2')
    output = tf.nn.bias_add(output,biases['conv5_2'])
    output = tf.nn.relu(output)
    
    output = fire_module(output,s1=16,e1=16,e3=16, channel=32,fire_id='fire11')
    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout2')
    
    output = tf.nn.conv2d(output,weights['conv5_3'], strides=[1,1,1,1], padding='SAME', name='conv5_3')
    output = tf.nn.bias_add(output,biases['conv5_3'])
    output = tf.nn.relu(output)
    output = tf.nn.conv2d(output,weights['conv5_4'], strides=[1,1,1,1], padding='SAME', name='conv5_4')
    output = tf.nn.bias_add(output,biases['conv5_4'])
    output = tf.nn.relu(output)
    output = tf.nn.conv2d(output,weights['conv5_5'], strides=[1,1,1,1], padding='SAME', name='conv5_5')
    output = tf.nn.bias_add(output,biases['conv5_5'])
    output = tf.nn.relu(output)
    
    output = tf.nn.avg_pool(output, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')
    
    output = tf.squeeze(output,[1,2])

    return output
total_iterations = 0
loss_train = []
acc_train = []
loss_test = []
acc_test_vr = []
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
       # x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        x_batch,y_true_batch = random_batch() 
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        _,loss = session.run([optimizer,cost], feed_dict=feed_dict_train)
        loss_train.append(loss)

        acc = session.run(accuracy, feed_dict=feed_dict_train)
        #msg1 = "Optimization Iteration: {0:>6}, Training loss: {1:>6.1%}"
        print('Loss: ' ,loss)
        print('number of iteration',i,'training accuracy:',acc*100,'%')
        acc_train.append(acc)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            x_batch_test,y_true_batch_test = random_batch_test() 
            feed_dict_train = {x: x_batch_test,
                           y_true: y_true_batch_test}
            
            _,loss_te = session.run([optimizer,cost], feed_dict=feed_dict_train)
            acc_test = session.run(accuracy, feed_dict=feed_dict_train)
            loss_test.append(loss_te)
            acc_test_vr.append(acc_test)
            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, test Accuracy: {1:>6.1%}"
            
            # Print it.
            print(msg.format(i + 1, acc_test))
            print('Loss_test:',loss_te)
            
        if i% 1000 == 0:
            save_path = saver.save(session, "Modelsqueeze_VGG/netVGG.ckpt")
            print("Model saved in path: %s" % save_path)

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


x = tf.placeholder(tf.float32, (None,img_size,img_size,num_channels),name='x')
#x_image = tf.reshape(x,[-1,img_size,img_size,3])

y_true = tf.placeholder(tf.float32,(None,10),name='y_true')
y_true_cls = tf.argmax(y_true,1)

logits = squeeze_netv_1_1(x,10)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred,1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
#print("loss: ", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(cost)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
saver = tf.train.Saver()
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 128

optimize(100001)

