'''
# Network Architecture of DCGAN
# Author: Zhihui Lu
# Date: 2019/08/02
'''

import tensorflow as tf
import numpy as np

def generator(input, shape=(7, 7, 128), is_training=True, drop_rate=0.7):
    x = tf.keras.layers.Dense(units = np.prod(shape), activation=tf.nn.leaky_relu)(input)
    x = batch_norm(x, is_training)
    x = tf.reshape(x , shape=[-1] + list(shape))

    x = deconv_with_bn(x, filters = 64, drop_rate=drop_rate, is_training=is_training)
    x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (3, 3), strides = (2, 2), padding = 'same',
                                        activation = tf.nn.sigmoid)(x)
    return x

def discriminator(input, drop_rate=0.7):
    x = conv_with_bn(input, filters=64, drop_rate=drop_rate)
    x = conv_with_bn(x, filters=128, drop_rate=drop_rate)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units = 1, activation = tf.nn.sigmoid)(x)
    return x

# convolution operation
def conv_with_bn(input, filters, kernel_size=(3, 3), strides=(2, 2), padding='same', drop_rate=0.7):
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    output = tf.nn.leaky_relu(output)
    output = tf.keras.layers.Dropout(drop_rate)(output)
    return output

# deconvolution operation
def deconv_with_bn(input, filters, kernel_size=(3, 3), strides=(2, 2), padding='same', is_training=True, drop_rate=0.7):
    output = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                             strides=strides, padding=padding)(input)
    output = tf.nn.leaky_relu(output)
    output = tf.keras.layers.Dropout(drop_rate)(output)
    output = batch_norm(output, is_training=is_training)
    return output


# batch normalization
def batch_norm(x, is_training=True):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training)
