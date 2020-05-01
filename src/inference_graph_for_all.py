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

output_node_names = ['conv12/bias_add']
input_node_names = ['image_input']

checkpoints = []
if 'model.ckpt' in FLAGS.train_dir:
	checkpoints.append(FLAGS.train_dir)
else:
	checkpoints.extend([os.path.join(x[0], 'model.ckpt-200000') for x in os.walk(FLAGS.train_dir) if 'inference' not in x[0] and 'model.ckpt-200000.index' in x[2]])
for i, c in enumerate(checkpoints):
	print("\nProcessing:", c.split('\\')[-2])
	if '4' in c and '-4' not in c:
		print("mask_parameters_now:", 4)
		mask_param = 4
	else:
		print("mask_parameters_now:", 8)
		mask_param = 8
	if '_lin_lin_anch' in c:
		print("encoding_type_now: asymmetric_linear")
		print('log_anchors_now:', False)
		encoding_scheme = 'asymmetric_linear'
		use_log_anchors = False
	elif '_log_log_anch' in c:
		print("encoding_type_now: asymmetric_log")
		print("log_anchors_now:", True)
		encoding_scheme = 'asymmetric_log'
		use_log_anchors = True
	else:
		print("encoding_type_now: normal")
		encoding_scheme = 'normal'
		if 'spatial' in c:
			print("log_anchors_now:", False)
			use_log_anchors = False
		else:
			print("log_anchors_now:", True)
			use_log_anchors = True

	if not tf.gfile.Exists(FLAGS.out_dir):
		tf.gfile.MakeDirs(FLAGS.out_dir)
	log_dir = os.path.join(FLAGS.out_dir, str(c.split('\\')[-2]))
	if not tf.gfile.Exists(log_dir):
		tf.gfile.MakeDirs(log_dir)

	with tf.Graph().as_default():
		mc = cityscape_squeezeDet_config(mask_param, use_log_anchors, False, encoding_scheme)
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
		# assert os.path.exists(c), "Invalid checkpoint! "+str(c)
		saver.restore(sess, c)
		tf.train.write_graph(sess.graph.as_graph_def(), log_dir, 'tensorflowModel.pbtxt', as_text=True)
		transforms = ['add_default_attributes',
							'remove_nodes(op=Identity, op=CheckNumerics)',
							'fold_batch_norms', 'fold_old_batch_norms',
							'strip_unused_nodes', 'sort_by_execution_order']
		transformed_graph_def = TransformGraph(sess.graph_def, input_node_names, output_node_names, transforms)
		frozen_graph_def = tf.graph_util.convert_variables_to_constants(
		sess,
		transformed_graph_def,
		output_node_names)

		with open(os.path.join(log_dir, 'frozen_inference_graph.pb'), 'wb') as f:
			f.write(frozen_graph_def.SerializeToString())
