import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
import scipy.io
from MODEL_EDSR import model
from skimage import color
#from MODEL_FACTORIZED import model_factorized
import time

input_dir = '../DIV2K_valid_LR_x8'
output_dir = './output2'

model_path = '/usr/project/xtmp/webster/EDSR_15_checkpoints/EDSR_const_clip_0.01_epoch_001.ckpt-64740'

def get_img_list():
	files = [x for x in os.listdir(input_dir) if x.endswith('.png') and not x.startswith('.')]
	return sorted(files)

def get_test_image(img_list, index):
	filename = img_list[index]
	path = os.path.join(input_dir, filename)
	img = Image.open(path)
	return np.array(img)

def run_sess(sess, img):
	shape = img.shape
	img = sess.run(train_output, feed_dict={input_tensor: img.reshape(1, img.shape[0], img.shape[1], 3)})
	img = img.reshape(shape[0]*8, shape[1]*8, 3)
	return img


def test_VDSR_with_sess():
	saver.restore(sess, model_path)
	img_list = get_img_list()
	for i in range(len(img_list)):
		img = get_test_image(img_list, i)

		img_1 = np.rot90(img)
		img_2 = np.rot90(img_1)
		img_3 = np.rot90(img_2)
		img_4 = np.flipud(img)
		img_5 = np.fliplr(img)
		img_6 = np.roll(img, 1, axis=2)
		img_7 = np.roll(img, 2, axis=2)
		
		
		output_0 = run_sess(sess, img)
		output_1 = np.rot90(run_sess(sess, img_1),3)
		output_2 = np.rot90(run_sess(sess, img_2),2)
		output_3 = np.rot90(run_sess(sess, img_3),1)
		output_4 = np.flipud(run_sess(sess, img_4))
		output_5 = np.fliplr(run_sess(sess, img_5))
		output_6 = np.roll(run_sess(sess, img_6), 2, axis=2)
		output_7 = np.roll(run_sess(sess, img_7), 1, axis=2)

		img = (output_0 + output_1 + output_2 + output_3 + output_4 + output_5 + output_6 + output_7)/8

		img = Image.fromarray(img.astype('uint8'))
		img.save(os.path.join(output_dir, img_list[i]), compress_level=0)


gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)

with tf.Session(config=config) as sess:
	input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3))
	target_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3))
	shared_model = tf.make_template('shared_model', model)
	train_output = shared_model(input_tensor, scale=8, feature_size=256, num_layers=32)
	saver = tf.train.Saver()
	initializer = tf.global_variables_initializer()
	sess.run(initializer)
	test_VDSR_with_sess()
