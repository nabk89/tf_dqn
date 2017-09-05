import random, time, os
import numpy as np
import tensorflow as tf

from replay_memory import *
from environment import *

class DQN(object):
	def __init__(self, args, session, replay_memory, environment):
		self.args = args
		self.sess = session
		self.memory = replay_memory
		self.env = environment
		self.num_actions = self.env.num_actions
		if self.args.simple:
			self.input_size = self.env.input_size
		else:
			self.input_size = [self.args.frame_height, self.args.frame_width, self.args.state_length]
		#self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[None], name='lr')
		self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

		self.states = tf.placeholder(tf.float32, [None] + self.input_size)
		self.actions = tf.placeholder(tf.int32, [None])
		self.max_q = tf.placeholder(tf.float32, [None])
		self.rewards = tf.placeholder(tf.float32, [None])
		self.terminals = tf.placeholder(tf.float32, [None])

		self.prediction_Q = self.build_network('pred')
		self.target_Q = self.build_network('target')

		self.loss, self.optimizer = self.build_optimizer()
                       
	def build_network(self, name):
		with tf.variable_scope(name):
			if self.args.simple:
				net = tf.layers.dense(inputs=self.states, units=10, activation=tf.nn.relu, kernel_initializer = self.initializer)
				#net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu, kernel_initializer=self.initializer)
				Q = tf.layers.dense(inputs=net, units=self.num_actions, activation=None, kernel_initializer=self.initializer)
			else:
				if self.args.nips:
					# Mnih et. al. (NIPS 2013)
					# Input image:      4 x 84 x 84 (4 gray-scale images of 84 x 84 pixels).
					# Conv layer 1:     16 filters 8 x 8, stride 4, relu.
					# Conv layer 2:     32 filters 4 x 4, stride 2, relu.
					# Fully-conn. 1:    256 units, relu.
					# Fully-conn. 2:    num-action units, linear.
					conv1 = tf.layers.conv2d(inputs=self.states, filters=16, kernel_size=[8,8], strides=[4,4], padding='VALID', activation=tf.nn.relu, kernel_initializer=self.initializer)
					conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=[2,2], padding='VALID', activation=tf.nn.relu, kernel_initializer=self.initializer)
					flatten = tf.contrib.layers.flatten(conv2)
					dense = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu, kernel_initializer=self.initializer)
					Q = tf.layers.dense(inputs=dense, units=self.num_actions, activation=None, kernel_initializer=self.initializer)
				else: 
					# Mnih et. al. (Nature 2015)
					# Input image:      4 x 84 x 84 (4 gray-scale images of 84 x 84 pixels).
					# Conv layer 1:     32 filters 8 x 8, stride 4, relu.
					# Conv layer 2:     64 filters 4 x 4, stride 2, relu.
					# Conv layer 3:     64 filters 3 x 3, stride 1, relu.
					# Fully-conn. 1:    512 units, relu. 
					# Fully-conn. 2:    num-action units, linear.
					conv1 = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[8,8], strides=[4,4], padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
					conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
					conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3,3], strides=[1,1], padding='SAME', activation=tf.nn.relu, kernel_initializer=self.initializer)
					flatten = tf.contrib.layers.flatten(conv3)
					dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu, kernel_initializer=self.initializer)
					Q = tf.layers.dense(inputs=dense, units=self.num_actions, activation=None, kernel_initializer=self.initializer)
			return Q
	
	def build_optimizer(self):
		# Calculate the target Q value (= r + gamme * maxQ(next state))
		self.target_q = self.rewards + tf.multiply(1-self.terminals, tf.multiply(self.args.discount_factor, self.max_q))
		
		# Calculate the predicted Q value
		action_one_hot = tf.one_hot(indices=self.actions, depth=self.num_actions, on_value=1.0, off_value=0.0)
		pred_q = tf.reduce_sum(tf.multiply(self.prediction_Q, action_one_hot), reduction_indices=1)
		# This is doubtful codes.... the dimension matches????
		
        # Calculate the loss and make an optimizer
		loss = tf.reduce_mean(tf.square(self.target_q - pred_q))
		if self.args.simple:
			optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(loss)
		else:
			optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate, decay=0.99, momentum=0, epsilon=1e-6).minimize(loss)
			# optimize = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
		return loss, optimizer
	
	def train_network(self):
		# Fetch a mini-batch from the replay memory
		current_state, action, reward, terminal, next_state = self.memory.mini_batch()

		# Calculate the target Q value (batch)
		next_q_batch = self.sess.run(self.target_Q, feed_dict={self.states: next_state})
		next_max_q_batch = np.max(next_q_batch, axis=1)

		# Run optimizer
		feeds = {self.states: current_state, self.actions: action, self.rewards: reward, self.terminals: terminal, self.max_q: next_max_q_batch}
		return self.sess.run([self.loss, self.optimizer], feed_dict=feeds)

	def update_target_network(self):
		copy_op = []
		pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
		for pred_var, target_var in zip(pred_vars, target_vars):
			copy_op.append(target_var.assign(pred_var.value()))
		self.sess.run(copy_op)

	def predict_Q(self, state):
		return self.sess.run(self.prediction_Q, feed_dict={self.states: np.reshape(state, [1] + self.input_size)})
