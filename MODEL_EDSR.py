import tensorflow as tf
import numpy as np

SCALING_FACTOR = 0.1

def resBlock(x, channels=256, kernel_size=[3,3], scale=1):
	tmp = tf.layers.conv2d(x, channels, kernel_size, activation=None, padding='SAME')
	tmp = tf.nn.relu(tmp)
	tmp = tf.layers.conv2d(tmp, channels, kernel_size, activation=None, padding='SAME')
	tmp *= scale
	return x + tmp

def _phase_shift(I, r):
	return tf.depth_to_space(I, r)

def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
	return X

def upsample(x, scale=8, features=256, activation=tf.nn.relu):
	assert scale in [2, 3, 4, 8]
	x = tf.layers.conv2d(x, features, [3,3], activation=activation, padding='SAME')
	if scale == 2:
		ps_features = 3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME') #Increase channel depth
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x, 2, color=True)
	elif scale == 3:
		ps_features  =3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x, 3, color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	elif scale == 8:
		ps_features = 3*(8*8)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		#x = tf.layers.conv2d_transpose(x,3,kernel_size=(3,3),strides=2,activation=activation, padding='SAME')
		x = PS(x, 8, color=True)
	return x

def model(input_tensor, scale=8, feature_size=256, num_layers=32):
	with tf.device("/gpu:0"):
		'''
		Preprocessing by subtracting the batch mean
		'''
		#input_mean = tf.reduce_mean(input_tensor
		input_mean = tf.reduce_mean(input_tensor, 2, keep_dims=True)
		input_mean = tf.reduce_mean(input_mean, 1, keep_dims=True)
		tensor = input_tensor - input_mean

		'''
		Conv 1
		'''
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME') #Linear activation by default
		conv_1 = tensor #backup

		#Building resBlocks
		for i in range(10):
			tensor = resBlock(tensor, feature_size, scale=SCALING_FACTOR)

		#Upsampling x2
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME')
		tensor += conv_1
		tensor = upsample(tensor, 2, feature_size, activation=None)

		'''
		Conv 2
		'''
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME') #Linear activation by default
		conv_1 = tensor #backup

		#Building resBlocks
		for i in range(10):
			tensor = resBlock(tensor, feature_size, scale=SCALING_FACTOR)

		#Upsampling x2
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME')
		tensor += conv_1
		tensor = upsample(tensor, 2, feature_size, activation=tf.nn.relu)

		'''
		Conv 3
		'''
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME') #Linear activation by default
		conv_1 = tensor #backup

		#Building resBlocks
		for i in range(10):
			tensor = resBlock(tensor, feature_size, scale=SCALING_FACTOR)

		#Upsampling x2
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME')
		tensor += conv_1
		tensor = upsample(tensor, 2, feature_size, activation=None)

		'''
		Final smoothing
		'''
		tensor = tf.layers.conv2d(tensor, feature_size/4, [3,3], padding='SAME') #Linear activation by default
		conv_1 = tensor #backup
		for i in range(2):
			tensor = resBlock(tensor, feature_size/4, scale=SCALING_FACTOR)
		tensor = tf.layers.conv2d(tensor+conv_1, 3, [3,3], padding='SAME') #Linear activation by default

		tensor = tf.clip_by_value(tensor+input_mean, 0.0, 255.0)

		return tensor
