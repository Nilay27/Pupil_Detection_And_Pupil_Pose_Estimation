import tensorflow as tf
import os
import h5py
import numpy as np
import time
import glob
from PIL import Image
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt 
import pickle


close_left_eye=glob.glob('dataset_B_Eye_Images/closedLeftEyes/*.jpg')
close_right_eye=glob.glob('dataset_B_Eye_Images/closedRightEyes/*.jpg')
open_left_eye=glob.glob('dataset_B_Eye_Images/openLeftEyes/*.jpg')
open_right_eye=glob.glob('dataset_B_Eye_Images/openRightEyes/*.jpg')

close_eye=close_left_eye+close_right_eye
open_eye=open_left_eye+open_right_eye


n_classes = 2 
batch_size = 32

x = tf.placeholder('float', [None, 24, 24])
y = tf.placeholder('float')

keep_rate = 0.8
hm_epochs = 100

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

def train_neural_network(x):
	output = convolutional_neural_network(x)
	prediction = tf.nn.softmax(output)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
	
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		imgs_close = np.stack([open_image(fn) for fn in close_eye])
		print imgs_close.shape

		labels_close=np.zeros(shape=(len(imgs_close),2))
		print labels_close.shape

		for i in range (len(imgs_close)):
			labels_close[i][0]=1
		

		imgs_open = np.stack([open_image(fn) for fn in open_eye])
		print imgs_open.shape
		labels_open=np.zeros(shape=(len(imgs_open),2))
		print labels_open.shape
		for i in range (len(imgs_open)):
			labels_open[i][1]=1
		imgs = np.concatenate((imgs_close,imgs_open))
		print imgs.shape
		labels = np.concatenate((labels_close, labels_open))
		print labels.shape
		codes=list(zip(imgs,labels))
		print len(codes)

		
		shuffle(codes)
		images=np.stack([codes[i][0] for i in range(len(codes))])
		labels=np.stack([codes[i][1] for i in range(len(codes))])
		print type(images)
		print images.shape
		print labels.shape
		images=images/255.
		av=np.mean(images)
		images=images-av

		train_images=images[:int(0.6*(len(images)))]
		train_labels=labels[:int(0.6*(len(labels)))]
		val_images=images[int(0.6*(len(images))):int(0.8*(len(images)))]
		val_labels=labels[int(0.6*(len(labels))):int(0.8*(len(labels)))]
		test_images=images[int(0.8*(len(images))):]
		test_labels=labels[int(0.8*(len(images))):]
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		val_plot=[]
		train_plot=[]
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			acc_list1=[]
			while(i<len(train_images)):				
				epoch_x, epoch_y = train_images[i:i+batch_size],train_labels[i:i+batch_size]
				_, c, batch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c/len(train_images)
				acc_list1.append(batch_acc)
				i=i+batch_size
			tot_train_acc=np.mean(acc_list1)
			print 'Training Accuracy: ',tot_train_acc
			acc_list=[]
			i=0
			while(i<len(val_images)):				
				epoch_x, epoch_y = val_images[i:i+batch_size],val_labels[i:i+batch_size]
				
				batch_acc = sess.run(accuracy,feed_dict={x: epoch_x, y: epoch_y})
				acc_list.append(batch_acc)
				i=i+batch_size
			tot_val_acc=np.mean(acc_list)
			print 'Validation Accuracy: ',tot_val_acc
			val_plot.append(tot_val_acc)			
			train_plot.append(tot_train_acc)
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		with open('plot.pkl', 'wb') as pfile:
			pickle.dump([train_plot, val_plot],file=pfile)	

		with open('model.pkl','wb') as pfile:
			pickle.dump(sess.run([weights, biases]),file=pfile)
			
		
		i=0
		while(i<len(test_images)):				
			epoch_x, epoch_y = val_images[i:i+batch_size],val_labels[i:i+batch_size]
			
			batch_acc = sess.run(accuracy,feed_dict={x: epoch_x, y: epoch_y})
			acc_list.append(batch_acc)
			i=i+batch_size
		tot_test_acc=np.mean(acc_list)
		print 'Testing Accuracy: ', tot_test_acc

train_neural_network(x)