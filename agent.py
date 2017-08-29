import random, time, os
import numpy as npi
from tqdm import tqdm
import tensorflow as tf

from dqn import *
from replay_memory import *
from environment import *

class Agent:
	def __init__(self, args, session):
		self.args = args
		if self.args.simple:
			self.env = SimpleEnvironment(args)
			self.memory = SimpleMemory(args, self.env)
		else:
			self.env = AtariEnvironment(args)
			self.memory = ReplayMemory(args)

		self.sess = session
		self.dqn = DQN(self.args, self.sess, self.memory, self.env)

		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()

	def train(self):
		episodes_count = 0
		episode_reward = 0
		episode_rewards = []
		total_reward = 0
		total_loss = 0

		self.eps = self.args.eps_init

		# Start the first episode
		print('Start Training...')
		# state = self.env.new_epdisode_random_start()
		state = self.env.new_episode()

		# Synchronize the target network with the main network
		self.dqn.update_target_network()

		for self.step in tqdm(range(1, self.args.max_step+1), ncols=70, initial=0):
			# Select an action by epsilon-greedy 
			action = self.select_action(state)
			
			if self.args.simple:
				next_state, reward, terminal = self.env.act(action)
				if terminal:
					reward = -1
				else:
					reward = 0.1
				self.memory.add(action, reward, terminal, next_state)
				state = next_state
			else:
				next_frame, reward, terminal = self.env.act(action)
				# Make a new state
				state[0:self.args.state_length-1] = state[1:self.args.state_length]
				state[self.args.state_length-1] = next_frame
				# Do reward-clipping
				# this code is not needed in BreakOut-v0
				#reward = max(self.args.min_reward, max(self.args.max_reward, reward))
				# Save trasition in the replay memory
				self.memory.add(action, reward, terminal, next_frame)

			# Periodically update main network and target network
			if self.step >= self.args.training_start_step:
				if self.step % self.args.train_freq == 0:
					loss, _ = self.dqn.train_network()
					total_loss += loss

				if self.step % self.args.copy_freq == 0:
					self.dqn.update_target_network()

				if self.step % self.args.save_freq == 0:
					self.save()
					#self.args.display = True
			
				#if self.step % 299 == 0:
				#	self.args.display = False

				if self.step % self.args.show_freq == 0:
					avg_r = np.mean(episode_rewards)
					max_r = np.max(episode_rewards)
					min_r = np.min(episode_rewards)
					print('\ntotal_loss: %f, avg_r: %.4f, max_r: %.4f, min_r: %.4f, steps: %d, episodes: %d'%(total_loss, avg_r, max_r, min_r, self.step, len(episode_rewards)))
					episode_rewards = []
					total_loss = 0
					
			if terminal:
				episodes_count += 1
				episode_rewards.append(episode_reward)
				episode_reward = 0
				# state = self.env.new_episode_rnadome_start()
				state = self.env.new_episode()
			else:
				episode_reward += reward
			'''
			if self.step >= self.args.training_start_step:
				if self.step % self.config.test_step == self.config.test_step -1:
					avg_reward = total_reward / self.config.test_step
					total_reward = 0
					max_reward = np.max(episode_rewards)
					min_reward = np.min(episode_rewards)
					mean_reward = np.mean(episode_rewards)
					episode_rewards = []
					avg_loss = total_loss / self.config.test_step
					total_loss = 0
					print('\navg_r: %.4f, avg_l: %0.6f, max_r: %.4f, min_r: %.4f, mean_r: %.4f, # episodes: %d'\
						% (avg_reward, avg_loss, max_reward, min_reward, mean_reward, episodes_count))
			'''
	
	def select_action(self, state, test_epsilon=None):
		if test_epsilon == None:
			#epsilon = max(self.args.eps_min, self.args.eps_init - self.step/self.args.max_step)
			temp_eps = self.eps - (self.args.eps_init - self.args.eps_min) / self.args.final_exploration_frame
			self.eps = max(temp_eps, self.args.eps_min)
		else:
			self.eps = test_epsilon
		if random.random() < self.eps:
			action = self.env.random_action()
		else:
			action = np.argmax(self.dqn.predict_Q(state))
		return action

	def play(self, num_step=500, num_episode=10, test_eps=None, render=False):
		if test_eps == None:
			#test_eps = self.args.eps_min
			test_eps = 0
		if not self.load():
			exit()

		best_reward, best_episode = 0, 0
		for episode in range(num_episode):
			#state = self.env.new_episode_random_start()
			state = self.env.new_episode()
			current_reward = 0

			for t in tqdm(range(num_step), ncols=70):
				# Select an action by epsilon-greedy 
				action = self.select_action(state, test_eps)

				# Act 
				next_frame, reward, terminal = self.env.act(action)

				# Update the state
				if self.args.simple:
					state = next_frame
					if terminal:
						break
					else:
						reward = 1
						current_reward += reward
				else:
					state[0:self.args.state_length-1] = state[1:self.args.state_length]
					state[self.args.state_length-1] = next_frame
					if terminal:
						break
					else:
						current_reward += reward
				
			if current_reward > best_reward:
				best_reward = current_reward
				best_episode = episode

			print("="*30)
			print("<%d> Current best reward: %d (%dth episode)" % (episode, best_reward, best_episode))
			print("="*30)

	@property
	def model_dir(self):
		return '{}_{}_batch'.format(self.args.env_name, self.args.batch_size)
		#return 'CartPole-v1_32_batch'

	def save(self):
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, str(self.step)))
		print('\nSave at %d steps' % self.step)

	def load(self):
		print('Loading checkpoint ...')
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
		if checkpoint_state and checkpoint_state.model_checkpoint_path:
			checkpoint_model = os.path.basename(checkpoint_state.model_checkpoint_path)
			self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
			print('Success to load %s' % checkpoint_model)
			return True
		else:
			print('Fail to find a checkpoint')
			return False
