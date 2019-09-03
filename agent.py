import tensorflow as tf
import numpy as np
import os
import random
from functools import partial
from network import A3C_Network
from utilities import *
from datetime import datetime
import time
import gym
import sys
from random import choice

# Each worker runs asynchronously and updates the global network
class A3C_Worker():
	def __init__(self, number, trainer, args):
		if number != 'test':
			self.name = "agent_" + str(number)
		else:
			self.name = "agent_test"
		self.number = number
		self.start_time = args.start_time
		self.model_path = args.model_path
		self.evaluate = args.evaluate
		self.discount_factor = args.gamma
		self.num_steps = args.num_steps
		self.num_actions = args.a_size

		#Setup Atari Environment
		self.env = atari_env(args.env, {'crop1': 2, 'crop2': 10, 'dimension2': 84}, args) # crop and resize info for MsPacman

		#Create and initialize local network with global network parameters
		self.local_net = A3C_Network(self.name, self.num_actions, trainer)
		self.update_local_net = self.local_net.update_network_op('global')

	def train(self, sess, thread_manager):
		print ("Starting worker " + str(self.number))
		action_indexes = np.arange(self.num_actions)
		done = True
		batch = Batch()
		tf.set_random_seed(1 + self.number)
		self.env.seed(1 + self.number)

		with sess.as_default(), sess.graph.as_default():
			while not thread_manager.should_stop():

				batch.reset()

				#Reset if end of episode
				if done:
					episode_reward = 0
					done = False
					last_lstm_state = self.local_net.lstm_state_init
					state = self.env.reset() #Begin new episode

				sess.run(self.update_local_net) #Initialize weights to global network
				batch.initial_feature = last_lstm_state

				for _ in range(self.num_steps):
					policy, value, last_lstm_state = sess.run([self.local_net.policy, self.local_net.value, self.local_net.state_out],
																	feed_dict={	self.local_net.inputs:[state],
																				self.local_net.state_in[0]:last_lstm_state[0],
																				self.local_net.state_in[1]:last_lstm_state[1]})
					
					value = value[0][0] #value is 2-dimensional
					action = np.random.choice(action_indexes, p = policy[0]) #pick action with given probability -> Converges fastest
					state, reward, done, _ = self.env.step(action) #Pick action

					batch.add_data(state, action, reward, value)

					episode_reward += reward #Increment for summary

					if done:
						break

				#Update master network
				if not done:
					bootstrap_value = sess.run(self.local_net.value,
									 feed_dict={self.local_net.inputs : [state],
												self.local_net.state_in[0] : last_lstm_state[0],
												self.local_net.state_in[1] : last_lstm_state[1]})
					batch.bootstrap = bootstrap_value[0][0]
				else:
					batch.bootstrap = 0.0

				#-----Calculate discounted rewards and advantage functionss-----------
				batch_rewards = np.array(batch.rewards)
				lstm_state = batch.initial_feature

				# Calculate discounted rewards
				batch_discounted_rewards = np.zeros([batch.size], np.float64)
				R = batch.bootstrap
				for i in reversed(range(batch.size)):
					R = batch_rewards[i] + self.discount_factor * R
					batch_discounted_rewards[i] = R

				# Calculate advantage
				batch_advantage = np.zeros([batch.size], np.float64)
				p_grad = batch.bootstrap
				for i in reversed(range(batch.size)):
					p_grad = batch_rewards[i] + self.discount_factor * p_grad
					batch_advantage[i] = p_grad - batch.values[i]

				# Update network parameters
				_, _ = sess.run([self.local_net.apply_grads, self.local_net.total_loss],
										feed_dict={	self.local_net.state_in[0]:lstm_state[0],
													self.local_net.state_in[1]:lstm_state[1],
													self.local_net.inputs	: batch.states,
													self.local_net.target_v	: batch_discounted_rewards,
													self.local_net.actions	: batch.actions,
													self.local_net.advantages: batch_advantage})

	def test(self, sess, thread_manager, saver, args):
		print('Test Agent')

		tf.set_random_seed(1)
		episode_reward = 0
		start_time = time.time()
		num_tests = 0
		reward_total_sum = 0
		last_lstm_state = self.local_net.lstm_state_init
		state = self.env.reset()
		flag = True
		max_score = 0
		if not self.evaluate:
			time.sleep(args.sleeptime)
		
		with sess.as_default(), sess.graph.as_default():
			while not thread_manager.should_stop():

				if flag and not self.evaluate:
					sess.run(self.update_local_net) #Initialzie weights to global network
					flag = False

				policy, last_lstm_state = sess.run([self.local_net.policy, self.local_net.state_out],
														feed_dict={	self.local_net.inputs:[state],
																	self.local_net.state_in[0]:last_lstm_state[0],
																	self.local_net.state_in[1]:last_lstm_state[1]})

				action = np.argmax(policy[0]) #pick action with highest probability
				state, reward, done, info = self.env.step(action) #Pick action
				episode_reward += reward

				if args.render:
					self.env.render()

				if done and not info:
					state = self.env.reset()
					last_lstm_state = self.local_net.lstm_state_init
				elif info:
					flag = True
					num_tests += 1
					reward_total_sum += episode_reward
					reward_mean = reward_total_sum / num_tests
					string_reward_mean = "{0:.4f}".format(reward_mean)

					#Print episode summary
					stop_time = time.time()
					time_elapsed = str(time.strftime("%H:%M:%S", time.gmtime(stop_time-start_time)))
					summary = str(datetime.now())[:-7]+"  |  Elapsed: "+time_elapsed+"  |  Count: "+str(num_tests)+"  |  Episode Reward: "+str(episode_reward)+"  |  Average Reward: "+string_reward_mean
					print(summary)
					if not self.evaluate:
						with open("log.txt", "a") as myfile:
							myfile.write(summary+"\n")

					if not self.evaluate and episode_reward > max_score:
						max_score = episode_reward
						print("Saving model...")
						with open("log.txt", "a") as myfile:
							myfile.write("Saving model...\n")
						saver.save(sess,self.model_path+'/model-'+str(max_score)+'.cptk')
						p = "Model successfully saved as " + str(max_score) + '.cptk'
						print(p)
						with open("log.txt", "a") as myfile:
							myfile.write(p + "\n")

					episode_reward = 0
					state = self.env.reset()
					last_lstm_state = self.local_net.lstm_state_init
					if not self.evaluate:
						time.sleep(args.sleeptime)