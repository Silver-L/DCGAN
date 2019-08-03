'''
# Network Architecture of DCGAN
# Author: Zhihui Lu
# Date: 2019/08/02
# tf.keras.layers will cause some problems !
# Using tf.layers instead of tf.keras.layers
'''

import tensorflow as tf
import numpy as np

def generator_layer(input, shape=(7, 7, 64), is_training=True, drop_rate=0.7):
    x = tf.layers.dense(input, units = np.prod(shape), activation=tf.nn.leaky_relu)
    x = tf.reshape(x , shape=[-1] + list(shape))

    x = deconv_with_bn(x, filters = 64, drop_rate=drop_rate, is_training=is_training)
    x = tf.layers.conv2d_transpose(x, filters = 1, kernel_size = 5, strides = 2, padding = 'same',
                                        activation = tf.nn.sigmoid)
    return x

def discriminator_layer(input, drop_rate=0.7):
    x = conv2d(input, kernel_size = 5, filters = 64, drop_rate = drop_rate)
    x = conv2d(x, kernel_size = 5, filters = 64, strides = 1, drop_rate = drop_rate)
    x = conv2d(x, kernel_size = 5, filters = 64, strides = 1, drop_rate = drop_rate)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.layers.dense(x, units = 128, activation = tf.nn.leaky_relu)
    x = tf.layers.dense(x, units = 1)
    return x

# convolution operation
def conv2d(input, filters, kernel_size=5, strides=2, padding='same', drop_rate=0.7):
    output = tf.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
                              activation = tf.nn.leaky_relu)(input)
    output = tf.layers.dropout(output, drop_rate)
    return output

# deconvolution operation
def deconv_with_bn(input, filters, kernel_size=3, strides=2, padding='same', is_training=True, drop_rate=0.7):
    output = tf.layers.conv2d_transpose(input, filters=filters, kernel_size=kernel_size,
                                             strides=strides, padding=padding, activation = tf.nn.leaky_relu)
    output = tf.layers.dropout(output, drop_rate)
    output = batch_norm(output, is_training=is_training)
    return output


# batch normalization
def batch_norm(x, is_training=True):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training)
