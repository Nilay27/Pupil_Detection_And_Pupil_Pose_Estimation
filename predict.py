import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import glob

n_classes = 2 
batch_size = 32
keep_rate = 0.8

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	#                        size of window         movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def open_image(fn): return np.array(Image.open(fn).resize((24,24), Image.NEAREST))


def convolutional_neural_network(x):
	global weights
	global biases
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
				'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
				'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
				'W_fc1':tf.Variable(tf.random_normal([3*3*128,1024])),
				'W_fc2':tf.Variable(tf.random_normal([1024,1024])),
				'out':tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
				'b_conv2':tf.Variable(tf.random_normal([64])),
				'b_conv3':tf.Variable(tf.random_normal([128])),
				'b_fc1':tf.Variable(tf.random_normal([1024])),
				'b_fc2':tf.Variable(tf.random_normal([1024])),
				'out':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1, 24, 24, 1])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
	
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
	conv3 = maxpool2d(conv3)

	fc1 = tf.reshape(conv3,[-1, 3*3*128])
	fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])
	fc1 = tf.nn.dropout(fc1, keep_rate)

	
	fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])
	fc2 = tf.nn.dropout(fc2, keep_rate)

	output = tf.matmul(fc2, weights['out'])+biases['out']

	return output

close_left_eye=glob.glob('dataset_B_Eye_Images/closedLeftEyes/*.jpg')
open_left_eye=glob.glob('dataset_B_Eye_Images/openLeftEyes/*.jpg')


x = tf.placeholder('float', [None, 24, 24])
y = tf.placeholder('float')
with tf.Session() as sess:
	out=convolutional_neural_network(x)
	pred=tf.nn.softmax(out)
	sess.run(tf.initialize_all_variables())
	with open('model.pkl','rb') as pfile:
		param=pickle.load(pfile)
	weights_list=['W_conv1','W_conv2','W_conv3','W_fc1','W_fc2','out']
	biases_list=['b_conv1','b_conv2','b_conv3','b_fc1','b_fc2','out']
	for w in weights_list:
		sess.run([weights[w].assign(param[0][w])])

	for b in biases_list:
		sess.run([biases[b].assign(param[1][b])])
	list_prediction=[]
	for input_image in open_left_eye:
		input_image = open_image(input_image).reshape(1,24,24)
		input_image=input_image/255.
		av=np.mean(input_image)
		input_image=input_image-av
		out1=sess.run(out, feed_dict={x: input_image})

		if(np.argmax(out1)==1):
			list_prediction.append(0)
		else:
			list_prediction.append(1)

	print list_prediction