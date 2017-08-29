import random, time, os
import numpy as np
import tensorflow as tf

from replay_memory import *
from environment import *

class DQN:
	def __init__(self, args, session, replay_memory, environment):
		self.args = args
		self.sess = session
		self.memory = replay_memory
		self.env = environment
		self.num_actions = self.env.num_actions
		if self.args.simple:
			self.input_size = self.env.input_size
		else:
			self.input_size = [self.args.state_length, self.args.frame_height, self.args.frame_width]
		#self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[None], name='lr')
		self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

		self.input_state = tf.placeholder(tf.float32, [None] + self.input_size, name='input_state')
		self.input_action = tf.placeholder(tf.int32, [None], name='input_action')
		self.input_y = tf.placeholder(tf.float32, [None])

		self.prediction_Q = self.build_network('pred')
		self.target_Q = self.build_network('target')

		self.loss, self.optimizer = self.build_optimizer()
                       
	def build_network(self, name):
		with tf.variable_scope(name):
			if self.args.simple:
				net = tf.layers.dense(inputs=self.input_state, units=10, activation=tf.nn.relu, kernel_initializer = self.initializer)
				#net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu, kernel_initializer=self.initializer)
				Q = tf.layers.dense(inputs=net, units=self.num_actions, activation=None, kernel_initializer=self.initializer)
				return Q
			else:
			# Mnih et. al. (2013)
			# Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
			# Conv layer 1:     16 filters 8 x 8, stride 4, relu.
			# Conv layer 2:     32 filters 4 x 4, stride 2, relu.
			# Fully-conn. 1:    256 units, relu.
			# Fully-conn. 2:    num-action units, linear.
        
			# Mnih et. al. (2015)
			# Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
			# Conv layer 1:     32 filters 8 x 8, stride 4, relu.
			# Conv layer 2:     64 filters 4 x 4, stride 2, relu.
			# Conv layer 3:     64 filters 3 x 3, stride 1, relu.
			# Fully-conn. 1:    512 units, relu. 
			# Fully-conn. 2:    num-action units, linear.
				conv1 = tf.layers.conv2d(inputs=self.input_state, filters=32, kernel_size=[8,8], strides=[4,4],
			                           padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
				conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4,4], strides=[2,2],
			                           padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
				conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3,3], strides=[1,1],
		    	                       padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
				flatten = tf.contrib.layers.flatten(conv3)
				dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu, kernel_initializer=self.initializer)
				Q = tf.layers.dense(inputs=dense, units=self.num_actions, activation=None, kernel_initializer=self.initializer)
				return Q
	
	def build_optimizer(self):
		# Calculate the predicted Q value
		action_one_hot = tf.one_hot(self.input_action, self.num_actions, 1.0, 0.0)
		pred_Q_t = tf.reduce_sum(tf.multiply(self.prediction_Q, action_one_hot), reduction_indices=1)

        # Calculate the loss and make an optimizer
		loss = tf.reduce_mean(tf.square(self.input_y - pred_Q_t))
		if self.args.simple:
			optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(loss)
		else:
			optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate, momentum=0.95).minimize(loss)
			# optimize = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
		return loss, optimizer
	
	def train_network(self):
		# Fetch a mini-batch from the replay memory
		current_state, action, reward, terminal, next_state = self.memory.mini_batch()

		# Calculate the target Q value 
		target_Q_t_plus_1 = self.sess.run(self.target_Q, feed_dict={self.input_state: next_state})
		target_Q_t = []
		for i in xrange(self.args.batch_size):
			if terminal[i]:
				target_Q_t.append(reward[i])
			else:
				target_Q_t.append(reward[i] + self.args.discount_factor * np.max(target_Q_t_plus_1[i]))

		return self.sess.run([self.loss, self.optimizer], feed_dict={self.input_state: current_state, 
															self.input_action: action, 
															self.input_y: target_Q_t})

	def update_target_network(self):
		copy_op = []
		pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
		for pred_var, target_var in zip(pred_vars, target_vars):
			copy_op.append(target_var.assign(pred_var.value()))
		self.sess.run(copy_op)

	def predict_Q(self, state):
		state_pred = np.empty([1] + self.input_size, np.float32)
		state_pred[0] = state
		return self.sess.run(self.prediction_Q, feed_dict={self.input_state: state_pred})
