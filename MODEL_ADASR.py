import tensorflow as tf
import numpy as np

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
	assert scale in [2, 3, 4, 8, 16, 32]
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
	elif scale == 16:
		ps_features = 3*(8*8)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		x = PS(x, 8, color=True)
		ps_features = 3*(2*2)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		x = PS(x, 2, color=True)
	elif scale == 32:
		ps_features = 3*(8*8)
		x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
		x = PS(x, 8, color=True)
		ps_features = 3*(2**2)
		for i in range(2):
			x = tf.layers.conv2d(x, ps_features, [3,3], activation=activation, padding='SAME')
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x, 2, color=True)
	return x

def EDSR_Block(tensor, scale=8, feature_size=256, num_layers=32, SCALING_FACTOR=0.1):
	'''
	First convolution layer, Feature extraction
	'''
	tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME') #Linear activation by default

	conv_1 = tensor #backup

	'''
	Building resBlocks
	'''
	for i in range(num_layers):
		tensor = resBlock(tensor, feature_size, scale=SCALING_FACTOR)

	'''
	Add back the conv_1 tensor
	'''
	tensor = tf.layers.conv2d(tensor, feature_size, [3,3], padding='SAME')
	tensor += conv_1
	
	'''
	Upsampling
	'''
	tensor = upsample(tensor, scale, feature_size, activation=None)

	return tensor


def model(input_tensor, scale=8, feature_size=256, num_layers=32):
	with tf.device("/gpu:2"):

		SCALING_FACTOR_input = tf.Variable(0.1, dtype=tf.float32)
		SCALING_FACTOR_down_2 = tf.Variable(0.1, dtype=tf.float32)
		SCALING_FACTOR_down_4 = tf.Variable(0.1, dtype=tf.float32)

		'''
		Preprocessing by subtracting the batch mean
		'''
		input_mean = tf.reduce_mean(input_tensor)
		tensor = input_tensor - input_mean

		'''
		Downsample
		'''
		tensor_down_2 = tf.layers.average_pooling2d(tensor, pool_size=[2,2], strides=2, padding='VALID')
		tensor_down_4 = tf.layers.average_pooling2d(tensor_down_2, pool_size=[2,2], strides=2, padding='VALID')

		'''
		Superresolution
		'''
		tensor_output = EDSR_Block(tensor, scale, feature_size, num_layers, SCALING_FACTOR=SCALING_FACTOR_input)
		tensor_output_down_2 = EDSR_Block(tensor_down_2, scale*2, feature_size, num_layers, SCALING_FACTOR=SCALING_FACTOR_down_2)
		tensor_output_down_4 = EDSR_Block(tensor_down_4, scale*4, feature_size, num_layers, SCALING_FACTOR=SCALING_FACTOR_down_4)

		'''
		Concatenation, should have 9 channels
		'''
		tensor_concat = tf.concat([tensor_output, tensor_output_down_2, tensor_output_down_4], axis=3)

		'''
		Downsample
		'''
		tensor_final_output = tf.layers.conv2d(tensor_concat, 3, [3,3], padding='SAME')

		tensor_final_output = tf.clip_by_value(tensor_final_output+input_mean, 0.0, 255.0)

		return tensor_final_output
