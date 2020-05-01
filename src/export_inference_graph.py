# Author: Fraunhofer IAIS (16/01/2020)

"""Code to export the checkpoint to frozen inference graph"""

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os
from nets import *
from config import *
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.tools.graph_transforms import TransformGraph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
		'train_dir', './data/model_checkpoints/squeezeDet/',
		"""Path to the model parameter directory.""")
tf.app.flags.DEFINE_string(
		'out_dir', './data/out/', """Directory to dump the frozen inference graph.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
							"""Neural net architecture. """)
tf.app.flags.DEFINE_boolean('log_anchors_now', False, """Use Log domain extracted anchors ?""")
tf.app.flags.DEFINE_string('encoding_type_now', 'normal',
							"""what type of encoding to use""")
tf.app.flags.DEFINE_integer('mask_parameterization_now', 4,
							"""Bounding box is 4, octagonal mask is 8. other values not supported""")

meta_path = FLAGS.train_dir+'/model.ckpt-200000'
output_node_names = ['conv12/bias_add']
input_node_names = ['image_input']

with tf.Graph().as_default():
	mc = cityscape_squeezeDet_config(FLAGS.mask_parameterization_now, FLAGS.log_anchors_now, False, FLAGS.encoding_type_now)
	mc.LOAD_PRETRAINED_MODEL = False
	mc.IS_TRAINING = False
	if FLAGS.net == 'squeezeDet':
		model = SqueezeDet_inf(mc)
	else:
		assert False, "Model not supported!"
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	saver = tf.train.Saver(tf.global_variables())

	init = tf.global_variables_initializer()
	sess.run(init)
	if os.path.exists(FLAGS.train_dir):
		saver.restore(sess, meta_path)
	else:
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		# if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
	tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.out_dir, 'tensorflowModel.pbtxt', as_text=True)
	transforms = ['add_default_attributes',
						'remove_nodes(op=Identity, op=CheckNumerics)',
						'fold_batch_norms', 'fold_old_batch_norms',
						'strip_unused_nodes', 'sort_by_execution_order']
	transformed_graph_def = TransformGraph(sess.graph_def, input_node_names, output_node_names, transforms)
	frozen_graph_def = tf.graph_util.convert_variables_to_constants(
		sess,
		transformed_graph_def,
		output_node_names)
	if not tf.gfile.Exists(FLAGS.out_dir):
		tf.gfile.MakeDirs(FLAGS.out_dir)
	with open(os.path.join(FLAGS.out_dir, 'frozen_inference_graph.pb'), 'wb') as f:
		f.write(frozen_graph_def.SerializeToString())
