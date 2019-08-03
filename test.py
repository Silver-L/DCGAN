'''
# DCGAN test
# Author: Zhihui Lu
# Date: 2019/08/03
'''

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

from tqdm import tqdm
from absl import flags, app

from network import *
from model import DCGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings

# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("indir", "", "directory of model")
flags.DEFINE_string("gpu_index", "0", "GPU-index")
flags.DEFINE_integer("noise_dim", 32, "dimension of noise")
flags.DEFINE_integer("batch_size", 256, "batch_size")
flags.DEFINE_integer("model_index", 30, "model index")
flags.DEFINE_list("image_size", [28, 28, 1], "image size")

def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:

        # set network
        kwars = {
            'sess': sess,
            'noise_dim': FLAGS.noise_dim,
            'image_size': FLAGS.image_size,
            'generator_layer': generator_layer,
            'discriminator_layer': discriminator_layer,
            'is_training': False
        }

        Model = DCGAN(**kwars)

        # initialize
        sess.run(init_op)

        # test
        Model.restore_model(os.path.join(FLAGS.indir,'model','model_{}'.format(FLAGS.model_index)))

        noise = np.random.uniform(-1., 1., size = [FLAGS.batch_size, FLAGS.noise_dim])
        samples = Model.gen_san(noise)

        samples = samples[:, :, :, 0]
        m = utils.montage(samples)
        gen_img = m
        plt.axis('off')
        plt.imshow(gen_img, cmap='gray')
        plt.show()

if __name__ == '__main__':
    app.run(main)