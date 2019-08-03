'''
# DCGAN Model
# Author: Zhihui Lu
# Date: 2019/08/02
'''

import os
import tensorflow as tf
import numpy as np

class DCGAN(object):

    def __init__(self, sess, noise_dim, image_size, generator_layer, discriminator_layer, is_training=True,
                 lr=1e-4):

        self._sess = sess
        self._noise_dim = noise_dim
        self._image_size = image_size
        self._is_training = is_training
        self._generator_layer = generator_layer
        self._discriminator_layer = discriminator_layer
        self._lr = lr

        self._build_graph()


    def _build_graph(self):
        with tf.variable_scope('input'):
            with tf.variable_scope('noise'):
                self._noise_input = tf.placeholder(tf.float32, shape=[None, self._noise_dim])

            with tf.variable_scope('image'):
                self._real_image_input = tf.placeholder(tf.float32, shape=[None] + list(self._image_size))

        # Build Generator
        self._gen_sample = self.generator(noise=self._noise_input)

        # Build Discriminator
        self._d_real = self.discriminator(image=self._real_image_input)
        self._d_fake = self.discriminator(image=self._gen_sample, reuse=True)

        # discriminator loss
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self._d_real, labels = tf.ones_like(self._d_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self._d_fake, labels = tf.zeros_like(self._d_fake)))
        self._d_loss = self.d_loss_fake + self.d_loss_real

        # generator loss
        self._g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self._d_fake, labels = tf.ones_like(self._d_fake)))

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.5, beta2=0.999)

        # parameters
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._train_d = optimizer.minimize(self._d_loss, var_list=self.d_vars)
            self._train_g = optimizer.minimize(self._g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)

    def generator(self, noise, shape=(7, 7, 128), is_training=True, reuse=False, drop_rate=0.7):
        with tf.variable_scope('Generator', reuse=reuse):
            output = self._generator_layer(input=noise, shape=shape, is_training=is_training, drop_rate=drop_rate)
            return output

    def discriminator(self, image, reuse=False, drop_rate=0.7):
        with tf.variable_scope('Discriminator', reuse=reuse):
            output = self._discriminator_layer(input=image, drop_rate=drop_rate)
            return output

    def update_g(self, noise):
        _, g_loss = self._sess.run([self._train_g, self._g_loss], feed_dict={self._noise_input: noise})
        return g_loss

    def update_d(self, noise, image):
        _, d_loss = self._sess.run([self._train_d, self._d_loss],
                                   feed_dict={self._noise_input: noise, self._real_image_input: image})
        return d_loss

    def gen_sample(self, noise):
        sample = self._sess.run(self._gen_sample, feed_dict={self._noise_input: noise})
        return sample

    def save_model(self, path, index):
        save = self.saver.save(self._sess, os.path.join(path, 'model' , 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)