import numpy as np
import re

from functools import wraps

has_tf = True
try:
	import tensorflow as tf
	slim = tf.contrib.slim
	trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
except ImportError:
	has_tf = False
	pass

def variable_scope_decorator(func):

	@wraps(func)
	def wrapper(self, var_name, *args, **kwargs):
		assert has_tf, "Please install Tensorflow!"
		with tf.variable_scope(var_name):
			return func(self, *args, **kwargs)

	return wrapper

class InceptionV3(object):

	class meta:
		input_size = 299
		n_conv_maps = 2048
		feature_size = 2048
		mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(3,1,1)

		classifier_layers = ["fc"]
		conv_map_layer = "body"
		feature_layer = "pool"


	def __init__(self,
		inputs,
		net_name="InceptionV3",
		min_depth=16,
		depth_multiplier=1.0,
		scope=None):
		"""
			Inception model from http://arxiv.org/abs/1512.00567.

			Note: originally from "slim.nets.inception_v3.py"

			Constructs an Inception v3 network from inputs to the given final endpoint.
			This method can const  n bnhgfruct the network up to the final inception block
			Mixed_7c.

			Note that the names of the layers in the paper do not correspond to the names
			of the endpoints registered by this function although they build the same
			network.

			Here is a mapping from the old_names to the new names:
			Old name          | New name
			=======================================
			conv0             | Conv2d_1a_3x3
			conv1             | Conv2d_2a_3x3
			conv2             | Conv2d_2b_3x3
			pool1             | MaxPool_3a_3x3
			conv3             | Conv2d_3b_1x1
			conv4             | Conv2d_4a_3x3
			pool2             | MaxPool_5a_3x3
			mixed_35x35x256a  | Mixed_5b
			mixed_35x35x288a  | Mixed_5c
			mixed_35x35x288b  | Mixed_5d
			mixed_17x17x768a  | Mixed_6a
			mixed_17x17x768b  | Mixed_6b
			mixed_17x17x768c  | Mixed_6c
			mixed_17x17x768d  | Mixed_6d
			mixed_17x17x768e  | Mixed_6e
			mixed_8x8x1280a   | Mixed_7a
			mixed_8x8x2048a   | Mixed_7b
			mixed_8x8x2048b   | Mixed_7c

			Args:
			inputs: a tensor of size [batch_size, height, width, channels].
			final_endpoint: specifies the endpoint to construct the network up to. It
				can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
				'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
				'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
				'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
			min_depth: Minimum depth value (number of channels) for all convolution ops.
				Enforced when depth_multiplier < 1, and not an active constraint when
				depth_multiplier >= 1.
			depth_multiplier: Float multiplier for the depth (number of channels)
				for all convolution ops. The value must be greater than zero. Typical
				usage will be to set this value in (0, 1) to reduce the number of
				parameters or computation cost of the model.
			scope: Optional variable_scope.

			Returns:
			tensor_out: output tensor corresponding to the final_endpoint.
			end_points: a set of activations for external use, for example summaries or
									losses.

			Raises:
			ValueError: if final_endpoint is not set to one of the predefined values,
									or depth_multiplier <= 0
		"""
		assert has_tf, "Please install Tensorflow!"

		super(InceptionV3, self).__init__()
		assert depth_multiplier > 0, \
			"depth_multiplier is not greater than zero."
		self.depth = lambda d: max(int(d * depth_multiplier), min_depth)
		self.inputs = inputs
		self.scope = scope

		with tf.variable_scope(scope, net_name, [self.inputs]):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
				# input 299 x 299 x 3
				self.head = net = self.init_head()
				# output 35 x 35 x 192

			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
				# input 35 x 35 x 192
				self.mixed_5b_values = values = self.block5('Mixed_5b', net, 32)
				self.mixed_5b = net = tf.concat(axis=3, values=values)
				# input 35 x 35 x 256
				self.mixed_5c_values = values = self.block5('Mixed_5c', net, 64, is_weird=True)
				self.mixed_5c = net = tf.concat(axis=3, values=values)
				# input 35 x 35 x 288
				self.mixed_5d_values = values = self.block5('Mixed_5d', net, 64)
				self.mixed_5d = net = tf.concat(axis=3, values=values)

				# input 35 x 35 x 288
				self.mixed_6a_values = values = self.block6a('Mixed_6a', net)
				self.mixed_6a = net = tf.concat(axis=3, values=values)
				# outuput 17 x 17 x 288

				# input 17 x 17 x 768
				self.mixed_6b_values = values = self.block6('Mixed_6b', net, 128)
				self.mixed_6b = net = tf.concat(axis=3, values=values)
				# input 17 x 17 x 768
				self.mixed_6c_values = values = self.block6('Mixed_6c', net, 160)
				self.mixed_6c = net = tf.concat(axis=3, values=values)
				# input 17 x 17 x 768
				self.mixed_6d_values = values = self.block6('Mixed_6d', net, 160)
				self.mixed_6d = net = tf.concat(axis=3, values=values)
				# input 17 x 17 x 768
				self.mixed_6e_values = values = self.block6('Mixed_6e', net, 192)
				self.mixed_6e = net = tf.concat(axis=3, values=values)

				# input 8 x 8 x 1280
				self.mixed_7a_values = values = self.block7a('Mixed_7a', net)
				self.mixed_7a = net = tf.concat(axis=3, values=values)
				# input 8 x 8 x 2048
				self.mixed_7b_values = values = self.block7('Mixed_7b', net, "b")
				self.mixed_7b = net = tf.concat(axis=3, values=values)
				# input 8 x 8 x 2048
				self.mixed_7c_values = values = self.block7('Mixed_7c', net, "c")
				self.mixed_7c = net = tf.concat(axis=3, values=values)

			self.feature = tf.reduce_mean(self.mixed_7c, [1, 2])


		config_sess = tf.ConfigProto(allow_soft_placement=True)
		config_sess.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config_sess)
		name_regex = re.compile(r"^{net_name}/(.*?):0$".format(net_name=net_name))

		self.params = {
			name_regex.search(p.name).group(1): p
				for p in slim.get_model_variables()}

	def init_head(self):
		# 299 x 299 x 3
		net = slim.conv2d(self.inputs,			self.depth(32), 	[3, 3], 	scope='Conv2d_1a_3x3', stride=2)
		self.head_conv0 = net
		# 149 x 149 x 32
		net = slim.conv2d(net,					self.depth(32), 	[3, 3], 	scope='Conv2d_2a_3x3')
		self.head_conv1 = net
		# 147 x 147 x 32
		net = slim.conv2d(net,					self.depth(64), 	[3, 3], 	scope='Conv2d_2b_3x3', padding='SAME')
		self.head_conv2 = net
		# 147 x 147 x 64
		net = slim.max_pool2d(net, 									[3, 3], 	scope='MaxPool_3a_3x3', stride=2)
		self.head_pool2 = net
		# 73 x 73 x 64
		net = slim.conv2d(net,					self.depth(80), 	[1, 1], 	scope='Conv2d_3b_1x1')
		self.head_conv3 = net
		# 73 x 73 x 80.
		net = slim.conv2d(net,					self.depth(192), 	[3, 3], 	scope='Conv2d_4a_3x3')
		self.head_conv4 = net
		# 71 x 71 x 192.
		net = slim.max_pool2d(net, 									[3, 3], 	scope='MaxPool_5a_3x3', stride=2)
		self.head_pool4 = net
		# 35 x 35 x 192.
		return net


	@variable_scope_decorator
	def block5(self, net, d, is_weird=False):
		with tf.variable_scope('Branch_0'):
			branch_0 = slim.conv2d(net, 		self.depth(64), 	[1, 1], 	scope='Conv2d_0a_1x1')

		with tf.variable_scope('Branch_1'):
			if is_weird:
				# for Mixed_5c names in this branch are messed up
				br1 = slim.conv2d(net, 			self.depth(48), 	[1, 1], 	scope='Conv2d_0b_1x1')
				branch_1 = slim.conv2d(br1, 	self.depth(64), 	[5, 5], 	scope='Conv_1_0c_5x5')
			else:
				br1 = slim.conv2d(net, 			self.depth(48), 	[1, 1], 	scope='Conv2d_0a_1x1')
				branch_1 = slim.conv2d(br1, 	self.depth(64), 	[5, 5], 	scope='Conv2d_0b_5x5')

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.conv2d(net, 		self.depth(64), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_2 = slim.conv2d(branch_2, 	self.depth(96), 	[3, 3], 	scope='Conv2d_0b_3x3')
			branch_2 = slim.conv2d(branch_2, 	self.depth(96), 	[3, 3], 	scope='Conv2d_0c_3x3')

		with tf.variable_scope('Branch_3'):
			branch_3 = slim.avg_pool2d(net, 						[3, 3], 	scope='AvgPool_0a_3x3')
			branch_3 = slim.conv2d(branch_3, 	self.depth(d), 		[1, 1], 	scope='Conv2d_0b_1x1')
		return [branch_0, branch_1, branch_2, branch_3]



	@variable_scope_decorator
	def block6a(self, net):

		with tf.variable_scope('Branch_0'):
			branch_0 = slim.conv2d(net, 		self.depth(384), 	[3, 3], 	scope='Conv2d_1a_1x1',  stride=2, padding='VALID')

		with tf.variable_scope('Branch_1'):
			branch_1 = slim.conv2d(net, 		self.depth(64), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_1 = slim.conv2d(branch_1, 	self.depth(96), 	[3, 3], 	scope='Conv2d_0b_3x3')
			branch_1 = slim.conv2d(branch_1, 	self.depth(96), 	[3, 3], 	scope='Conv2d_1a_1x1',  stride=2, padding='VALID')

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.max_pool2d(net, 						[3, 3], 	scope='MaxPool_1a_3x3', stride=2, padding='VALID')

		return [branch_0, branch_1, branch_2]


	@variable_scope_decorator
	def block6(self, net, d):

		with tf.variable_scope('Branch_0'):
			branch_0 = slim.conv2d(net, 		self.depth(192), 	[1, 1], 	scope='Conv2d_0a_1x1')

		with tf.variable_scope('Branch_1'):
			branch_1 = slim.conv2d(net, 		self.depth(d), 		[1, 1], 	scope='Conv2d_0a_1x1')
			branch_1 = slim.conv2d(branch_1, 	self.depth(d), 		[1, 7], 	scope='Conv2d_0b_1x7')
			branch_1 = slim.conv2d(branch_1, 	self.depth(192), 	[7, 1], 	scope='Conv2d_0c_7x1')

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.conv2d(net, 		self.depth(d), 		[1, 1], 	scope='Conv2d_0a_1x1')
			branch_2 = slim.conv2d(branch_2, 	self.depth(d), 		[7, 1], 	scope='Conv2d_0b_7x1')
			branch_2 = slim.conv2d(branch_2, 	self.depth(d), 		[1, 7], 	scope='Conv2d_0c_1x7')
			branch_2 = slim.conv2d(branch_2, 	self.depth(d), 		[7, 1], 	scope='Conv2d_0d_7x1')
			branch_2 = slim.conv2d(branch_2, 	self.depth(192), 	[1, 7], 	scope='Conv2d_0e_1x7')

		with tf.variable_scope('Branch_3'):
			branch_3 = slim.avg_pool2d(net, 						[3, 3], 	scope='AvgPool_0a_3x3')
			branch_3 = slim.conv2d(branch_3, 	self.depth(192), 	[1, 1], 	scope='Conv2d_0b_1x1')

		return [branch_0, branch_1, branch_2, branch_3]


	@variable_scope_decorator
	def block7a(self, net):

		with tf.variable_scope('Branch_0'):
			branch_0 = slim.conv2d(net, 		self.depth(192), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_0 = slim.conv2d(branch_0, 	self.depth(320), 	[3, 3], 	scope='Conv2d_1a_3x3', stride=2, padding='VALID')

		with tf.variable_scope('Branch_1'):
			branch_1 = slim.conv2d(net, 		self.depth(192), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_1 = slim.conv2d(branch_1, 	self.depth(192), 	[1, 7], 	scope='Conv2d_0b_1x7')
			branch_1 = slim.conv2d(branch_1, 	self.depth(192), 	[7, 1], 	scope='Conv2d_0c_7x1')
			branch_1 = slim.conv2d(branch_1, 	self.depth(192), 	[3, 3], 	scope='Conv2d_1a_3x3', stride=2, padding='VALID')

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.max_pool2d(net, 						[3, 3], 	scope='MaxPool_1a_3x3', stride=2, padding='VALID')

		return [branch_0, branch_1, branch_2]


	@variable_scope_decorator
	def block7(self, net, suffix):

		with tf.variable_scope('Branch_0'):
			branch_0 = slim.conv2d(net, 		self.depth(320), 	[1, 1], 	scope='Conv2d_0a_1x1')

		with tf.variable_scope('Branch_1'):
			branch_1 = slim.conv2d(net, 		self.depth(384), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_1 = tf.concat(axis=3, values=[
				slim.conv2d(branch_1, 			self.depth(384), 	[1, 3], 	scope='Conv2d_0b_1x3'),
				slim.conv2d(branch_1, 			self.depth(384), 	[3, 1], 	scope='Conv2d_0{}_3x1'.format(suffix)) # < weird convention in slim
			])

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.conv2d(net, 		self.depth(448), 	[1, 1], 	scope='Conv2d_0a_1x1')
			branch_2 = slim.conv2d(branch_2, 	self.depth(384), 	[3, 3], 	scope='Conv2d_0b_3x3')
			branch_2 = tf.concat(axis=3, values=[
				slim.conv2d(branch_2, 			self.depth(384), 	[1, 3], 	scope='Conv2d_0c_1x3'),
				slim.conv2d(branch_2, 			self.depth(384), 	[3, 1], 	scope='Conv2d_0d_3x1')
			])

		with tf.variable_scope('Branch_3'):
			branch_3 = slim.avg_pool2d(net, 						[3, 3], 	scope='AvgPool_0a_3x3')
			branch_3 = slim.conv2d(branch_3, 	self.depth(192), 	[1, 1], 	scope='Conv2d_0b_1x1')

		return [branch_0, branch_1, branch_2, branch_3]

	@classmethod
	def new(cls, checkpoints_path, train=False, arg_scope=None, graph=None, *args, **kwargs):
		graph = graph or tf.Graph()
		with graph.as_default():
			arg_scope = arg_scope or inception_arg_scope()

			inputs = tf.placeholder(tf.float32, [None, None, None, 3])
			with slim.arg_scope(arg_scope), \
				slim.arg_scope([slim.batch_norm, slim.dropout], is_training=train), \
				tf.variable_scope("InceptionV3", reuse=None) as scope:
				model = cls(
					inputs=inputs,
					scope=scope,
					*args, **kwargs)

			saver = tf.train.Saver()
			saver.restore(model.sess, checkpoints_path)
			return model

	def __call__(self, X):
		return self.sess.run(self.feature, feed_dict={self.inputs: X})

def inception_arg_scope(weight_decay=0.00004,
	use_batch_norm=True,
	batch_norm_decay=0.9997,
	batch_norm_epsilon=0.001,
	activation_fn=None,
	batch_norm_updates_collections=None):
	"""Defines the default arg scope for inception models.

	Args:
		weight_decay: The weight decay to use for regularizing the model.
		use_batch_norm: "If `True`, batch_norm is applied after each convolution.
		batch_norm_decay: Decay for batch norm moving average.
		batch_norm_epsilon: Small float added to variance to avoid dividing by zero
			in batch norm.
		activation_fn: Activation function for conv2d.
		batch_norm_updates_collections: Collection for the update ops for
			batch norm.

	Returns:
		An `arg_scope` to use for the inception models.
	"""
	assert has_tf, "Please install Tensorflow!"
	activation_fn = activation_fn or tf.nn.relu
	batch_norm_updates_collections = batch_norm_updates_collections or tf.GraphKeys.UPDATE_OPS
	if use_batch_norm:
		normalizer_fn = slim.batch_norm
		normalizer_params = {
			# Decay for the moving averages.
			'decay': batch_norm_decay,
			# epsilon to prevent 0s in variance.
			'epsilon': batch_norm_epsilon,
			# collection containing update_ops.
			'updates_collections': batch_norm_updates_collections,
			# use fused batch norm if possible.
			'fused': None,
	}
	else:
		normalizer_fn = None
		normalizer_params = {}

	# Set weight_decay for weights in Conv and FC layers.
	with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
		with slim.arg_scope([slim.conv2d],
				weights_initializer=slim.variance_scaling_initializer(),
				activation_fn=activation_fn,
				normalizer_fn=normalizer_fn,
				normalizer_params=normalizer_params) as sc:
			return sc
