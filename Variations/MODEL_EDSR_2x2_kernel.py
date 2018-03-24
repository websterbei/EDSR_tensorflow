import tensorflow as tf
import numpy as np

SCALING_FACTOR = 0.1

def resBlock(x, channels=256, kernel_size=[3,3], scale=0.1):
	tmp = tf.layers.conv2d(x, channels, kernel_size, activation=None, padding='same')
	tmp = tf.nn.relu(tmp)
	tmp = tf.layers.conv2d(tmp, channels, kernel_size, activation=None, padding='same')
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
	x = tf.layers.conv2d(x, features, [3,3], activation=activation, padding='same')
	if scale == 2:
		ps_features = 3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='same') #Increase channel depth
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x, 2, color=True)
	elif scale == 3:
		ps_features  =3*(scale**2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='same')
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x, 3, color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='same')
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	elif scale == 8:
		ps_features = 3*(2*2)
		for i in range(3):
			x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='same')
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	return x

def model(input_tensor, scale=8, feature_size=256, num_layers=32):
	with tf.device("/gpu:0"):
		'''
		Preprocessing by subtracting the batch mean
		'''
		input_mean = tf.reduce_mean(input_tensor)
		tensor = input_tensor - input_mean

		'''
		First convolution layer
		'''
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='same') #Linear activation by default

		conv_1 = tensor #backup

		'''
		Building resBlocks
		'''
		for i in range(num_layers):
			#tensor_kernel_3 = resBlock(tensor, feature_size, kernel_size=[3,3], scale=SCALING_FACTOR)
			#tensor_kernel_4 = resBlock(tensor, feature_size, kernel_size=[4,4], scale=SCALING_FACTOR)
			#tensor_kernel_5 = resBlock(tensor, feature_size, kernel_size=[5,5], scale=SCALING_FACTOR)
			#tensor = tf.concat([tensor_kernel_3, tensor_kernel_4, tensor_kernel_5], axis=3) #similar to Inception model
			#tensor = tf.layers.conv2d(tensor, feature_size, [1,1], activation=None, padding='same')
			tensor = resBlock(tensor, feature_size, kernel_size=[2,2], scale=SCALING_FACTOR)

		'''
		Add back the conv_1 tensor
		'''
		tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='same')
		tensor += conv_1

		'''
		Upsampling
		'''
		tensor = upsample(tensor, scale, feature_size, activation=None)

		tensor = tf.clip_by_value(tensor+input_mean, 0.0, 255.0)
		return tensor
