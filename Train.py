import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL_EDSR import model

DATA_PATH = "/usr/project/xtmp/yb44/EDSR_HR_train_40"
#DATA_PATH = "data/train"
INPUT_IMG_SIZE = (40,40)
OUTPUT_IMG_SIZE = (320, 320)
BATCH_SIZE = 20
BASE_LR = 0.0004
DECAY_RATE = 0.96
DECAY_STEP = 10000
MAX_EPOCH = 120

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	l = [f for f in l if os.path.basename(f).endswith('.mat')]
	print len(l)
	return l

def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	for mat in target_list:
		data = scipy.io.loadmat(mat)
		input_img = data['input_patch']
		gt_img = data['target_patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, INPUT_IMG_SIZE[1], INPUT_IMG_SIZE[0], 3])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, OUTPUT_IMG_SIZE[1], OUTPUT_IMG_SIZE[0], 3])
	return input_list, gt_list

if __name__ == '__main__':
    train_list = get_train_list(DATA_PATH)

    train_input  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1], 3))
    train_gt  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], 3))
	#output_edges  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], 3))
	#target_edges  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1], 3))

    shared_model = tf.make_template('shared_model', model)
    #train_output, output_edges, target_edges = shared_model(train_input, train_gt, scale=8, feature_size=256, num_layers=32)
    train_output = shared_model(train_input, scale=8, feature_size=256, num_layers=32)
    #loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
    loss_1 = tf.reduce_mean(tf.losses.absolute_difference(train_output, train_gt))
	#loss_2 = tf.reduce_mean(tf.losses.absolute_difference(output_edges, target_edges))
    loss = loss_1
    tf.summary.scalar("loss", loss)

    mse = tf.reduce_mean(tf.squared_difference(train_output, train_gt))
    PSNR = tf.constant(255**2,dtype=tf.float32)/mse
    PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

    global_step 	= tf.Variable(0, trainable=False)
    learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, DECAY_STEP, DECAY_RATE, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    shuffle(train_list)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('logs', sess.graph)

        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        if model_path:
        	print "restore model..."
        	saver.restore(sess, model_path)
        	assign_op = global_step.assign_op(0)
        	sess.run(assign_op)
        	print "Done"

        saved = False
        log_file = open('log.csv', 'w')
        counter = 0
        for epoch in xrange(0, MAX_EPOCH):
        	for step in range(len(train_list)//BATCH_SIZE):
        		offset = step*BATCH_SIZE
        		input_data, gt_data = get_image_batch(train_list, offset, BATCH_SIZE)
        		feed_dict = {train_input: input_data, train_gt: gt_data}
        		_,l,output,lr, g_step, mse_val, psnr_val = sess.run([opt, loss, train_output, learning_rate, global_step, mse, PSNR], feed_dict=feed_dict)
        		print "[epoch %2.4f] loss %.4f\tmse %.4f\tpsnr %.4f\tlr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, mse_val, psnr_val, lr)
        		log_file.write('%2.4f,%.4f,%.4f' % (epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, psnr_val) + '\n')
        		log_file.flush()
        		del input_data, gt_data
        		if not saved:
        			saver.save(sess, "/usr/project/xtmp/webster/EDSR_15_checkpoints/initial.ckpt")
        			saved = True

        	    if int(counter/0.25)>counter:
                    counter = int(counter/0.25)
                    saver.save(sess, "/usr/project/xtmp/webster/EDSR_15_checkpoints/EDSR_const_clip_0.01_epoch_%03d.ckpt" % counter ,global_step=global_step)
