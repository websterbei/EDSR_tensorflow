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

input_dir = '../DIV2K_test_LR_x8'
output_dir = './output'

model_path = './checkpoints/EDSR_const_clip_0.01_epoch_003.ckpt-113297'

def get_img_list():
	files = [x for x in os.listdir(input_dir) if x.endswith('.png') and not x.startswith('.')]
	return sorted(files)

def get_test_image(img_list, index):
	filename = img_list[index]
	path = os.path.join(input_dir, filename)
	img = Image.open(path)
	return np.array(img)

def test_VDSR_with_sess():
	saver.restore(sess, model_path)
	img_list = get_img_list()
	for i in range(len(img_list)):
		img = get_test_image(img_list, i)
		shape = img.shape
			
		ckpt_path = model_path
                target = np.zeros((img.shape[0]*8,img.shape[1]*8, 3))

                img = sess.run(train_output, feed_dict={input_tensor: img.reshape(1, img.shape[0], img.shape[1], 3), target_tensor: target.reshape(1, target.shape[0], target.shape[1], 3)})
		
		img = img.reshape(shape[0]*8, shape[1]*8, 3)
		img = Image.fromarray(img.astype('uint8'))
		img.save(os.path.join(output_dir, img_list[i]), compress_level=0)


with tf.Session() as sess:
	input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        target_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3))
	shared_model = tf.make_template('shared_model', model)
	train_output, output_edge, target_edge = shared_model(input_tensor, target_tensor, scale=8, feature_size=256, num_layers=32)
	saver = tf.train.Saver()
	initializer = tf.global_variables_initializer()
	sess.run(initializer)
	test_VDSR_with_sess()
