# Author: Fraunhofer IAIS (16/01/2020)

"""Code to export the checkpoint to frozen inference graph"""

import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'pb_file', './data/model_checkpoints/squeezeDet/',
    """Path to the .pb file""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump the tensorboard file for visualization.""")

with tf.Session() as sess:
    model_filename =FLAGS.pb_file
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

LOGDIR=FLAGS.out_dir
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()