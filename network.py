from __future__ import division
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
from utilities import normalized_columns_initializer

class A3C_Network():
	def __init__(self, scope, a_size, trainer):
		self.scope = scope
		with tf.variable_scope(scope):
			# Input layer
			self.inputs = tf.placeholder("float", [None, 80, 80, 1], name="inputs")

			# Layer 1
			self.conv1 = tf.layers.conv2d(
				inputs = self.inputs,
				filters = 32,
				kernel_size = [5,5],
				padding = "same")
			self.pool1 = tf.layers.max_pooling2d(inputs = self.conv1,
				pool_size = [2,2], strides = 2)

			# Layer 2
			self.conv2 = tf.layers.conv2d(
				inputs = self.pool1,
				filters = 32,
				kernel_size = [5,5],
				padding = "same",
				activation = tf.nn.relu)
			self.pool2 = tf.layers.max_pooling2d(inputs = self.conv2,
				pool_size = [2,2], strides = 2)

			# Layer 3
			self.conv3 = tf.layers.conv2d(
				inputs = self.pool2,
				filters = 64,
				kernel_size = [4,4],
				padding = "same",
				activation = tf.nn.relu)
			self.pool3 = tf.layers.max_pooling2d(inputs = self.conv3,
				pool_size = [2,2], strides = 2)

			h = slim.flatten(self.pool3)
			h = layer.fully_connected(h, 512, activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.0))
			
			# Fully Connected layers feed into LSTM
			lstm_input = tf.expand_dims(h, [0])
			step_size = tf.shape(self.inputs)[:1]

			lstm_cell = tf.contrib.rnn.BasicLSTMCell(512)
			c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
			h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
			self.lstm_state_init = [c_init, h_init]	# Initial state of LSTM - all zeros

			c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
			h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
			self.state_in = [c_in, h_in]
			lstm_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

			lstm_out, lstm_state_out = tf.nn.dynamic_rnn(	lstm_cell, lstm_input,
															initial_state=lstm_state_in,
															sequence_length=step_size)

			lstm_c, lstm_h = lstm_state_out
			self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

			lstm_out = tf.reshape(lstm_out, [-1, 512])
			self.policy = layer.fully_connected(lstm_out, a_size, activation_fn=tf.nn.softmax,
												weights_initializer=normalized_columns_initializer(0.01),
												biases_initializer=tf.constant_initializer(0))
			self.value = layer.fully_connected(	lstm_out, 1, activation_fn=None,
												weights_initializer=normalized_columns_initializer(1.0),
												biases_initializer=tf.constant_initializer(0))

			# If worker agent, implement gradient descent
			if scope != 'global' and scope != 'agent_test': #not global or test
				self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
				self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None],dtype=tf.float32, name="discounted_reward")
				self.advantages = tf.placeholder(shape=[None],dtype=tf.float32, name="advantage")

				v = tf.reshape(self.value, [-1])
				log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))
				responsible_outputs = tf.reduce_sum(tf.multiply(log_policy, self.actions_onehot), reduction_indices=1)

				# Loss functions
				policy_loss = -tf.reduce_sum(responsible_outputs*self.advantages)
				value_loss = 0.5 * tf.reduce_sum(tf.square(v - self.target_v))
				entropy = - tf.reduce_sum(self.policy * log_policy)
				self.total_loss = 0.5 * value_loss + policy_loss - entropy * 0.01

				# Get gradients from local network using local losses
				local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
				gradients = tf.gradients(self.total_loss, local_params)
				grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

				# Apply local gradients to global network
				global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(list(zip(grads, global_params)))

	def update_network_op(self, from_scope):
		with tf.variable_scope(self.scope):
			to_scope = self.scope
			from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
			to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

			ops = []
			for from_var,to_var in zip(from_vars,to_vars):
				ops.append(to_var.assign(from_var))

			return tf.group(*ops)
