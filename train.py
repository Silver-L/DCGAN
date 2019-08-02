'''
# DCGAN
# Author: Zhihui Lu
# Date: 2019/08/02
'''

import os
import tensorflow as tf
import numpy as np
import utils

from tqdm import tqdm
from absl import flags, app

from network import *
from model import DCGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("outdir", "H:/experiment_result/mnist/DCGAN", "output directory")
flags.DEFINE_string("gpu_index", "0", "GPU-index")
flags.DEFINE_integer("noise_dim", 3, "dimension of noise")
flags.DEFINE_integer("batch_size", 256, "batch_size")
flags.DEFINE_integer("epoch", 1000, "number of epoches")
flags.DEFINE_float("lr", 1e-4, "learning rate")
flags.DEFINE_list("image_size", [28, 28, 1], "image size")

def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # check folder
    if not os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard')):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))
    if not os.path.exists(os.path.join(FLAGS.outdir, 'model')):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)

    # preprocess
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:

        # set network
        kwars = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'noise_dim': FLAGS.noise_dim,
            'image_size': FLAGS.image_size,
            'generator_layer': generator_layer,
            'discriminator_layer': discriminator_layer,
            'is_training': True
        }

        Model = DCGAN(**kwars)

        utils.cal_parameter()

        # prepare tensorboard
        writer_train_g = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train_g'), sess.graph)
        writer_train_d = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train_d'))

        g_loss_value = tf.Variable(0.)
        d_loss_value = tf.Variable(0.)
        tf.summary.scalar("Generator", g_loss_value)
        tf.summary.scalar("Discriminator", d_loss_value)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        tbar = tqdm(range(FLAGS.epoch), ascii=True)
        epoch_loss_g = []
        epoch_loss_d = []

        for step in tbar:
            train_data_shuffled = train_gen.flow(x_train, y=None, batch_size=FLAGS.batch_size, shuffle=True)

            # one epoch
            for iter in range(x_train.shape[0] // FLAGS.batch_size):
                train_data_batch = next(train_data_shuffled)

                noise = np.random.normal(0., 1., size=[FLAGS.batch_size, FLAGS.noise_dim])
                # training
                d_loss = Model.update_d(noise, train_data_batch)
                g_loss = Model.update_g(noise)
                g_loss = Model.update_g(noise)

                epoch_loss_d.append(d_loss)
                epoch_loss_g.append(g_loss)

            s = "epoch:{}, loss_d:{:.4f}, loss_g:{:.4f}".format(step, np.mean(epoch_loss_d), np.mean(epoch_loss_g))
            tbar.set_description(s)

            sum_d_loss = sess.run(merge_op, {d_loss_value: np.mean(epoch_loss_d)})
            sum_g_loss = sess.run(merge_op, {g_loss_value: np.mean(epoch_loss_g)})
            writer_train_d.add_summary(sum_d_loss, step)
            writer_train_g.add_summary(sum_g_loss, step)

            epoch_loss_d.clear()
            epoch_loss_g.clear()

            # save model
            Model.save_model(step)

if __name__ == '__main__':
    app.run(main)
