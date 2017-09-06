import os, time
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from dqn import *
from replay_memory import *
from environment import *

class Agent(object):
	def __init__(self, args, session):
		self.args = args
		self.sess = session
		self.dqn = DQN(self.args, self.sess, self.memory, self.env)

		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		
		# Synchronize the target network with the main network
		self.dqn.update_target_network()

		# Seed of the random number generator
		np.random.seed(int(time.time()))

	def train(self):
		episodes_count = 0
		best_reward = 0
		episode_reward = 0
		episode_rewards = []
		episode_q = 0
		total_loss = 0
		self.action_histogram = [0] * self.env.num_actions
		self.random_histogram = [0] * self.env.num_actions
		print('Start Training...')

		self.reset_episode()
		for self.step in xrange(self.args.max_step):
		#for self.step in tqdm(range(1, self.args.max_step+1), ncols=70, initial=0):
			# Select an action by epsilon-greedy 
			action = self.select_action(self.state)
			if not self.args.simple:
				q = self.dqn.predict_Q_value(self.state)[0]
				q = q[action]
				episode_q += q
		
			next_frame, reward, terminal = self.env.act(action)
			self.memory.add(action, reward, terminal, next_frame)
			self.process_state(next_frame)

			episode_reward += reward
			if terminal:
				episodes_count += 1
				episode_rewards.append(episode_reward)
				if episode_reward > best_reward:
					best_reward = episode_reward
				episode_reward = 0
				self.reset_episode()

			# Periodically update main network and target network
			if self.step >= self.args.training_start_step:
				if self.step % self.args.train_freq == 0:
					loss, _ = self.dqn.train_network()
					total_loss += loss

				if self.step % self.args.copy_freq == 0:
					self.dqn.update_target_network()

				if self.step % self.args.save_freq == 0:
					self.save()
				
				'''
				if self.step % self.args.show_freq == 0:
					avg_r = np.mean(episode_rewards)
					max_r = np.max(episode_rewards)
					min_r = np.min(episode_rewards)
					print('\ntotal_loss: %f, avg_r: %.4f, max_r: %.4f, min_r: %.4f, # of episodes: %d'%(total_loss, avg_r, max_r, min_r, len(episode_rewards)))
					episode_rewards = []
					total_loss = 0
				'''
				if terminal:
					print('[%d ep, %d step] loss: %.4f, q: %.4f, reward: %d (best: %d)'\
						%(episodes_count, self.step, total_loss, episode_q, episode_rewards[-1], best_reward)\
						+ str(self.random_histogram) + " " + str(self.action_histogram))
					total_loss = 0
					episode_q = 0
					self.random_histogram = [0] * self.env.num_actions
					self.action_histogram = [0] * self.env.num_actions
		
	def play(self, num_episode=10):			
		if not self.load():
			exit()

		best_reward = 0
		for episode in range(num_episode):
			self.reset_episode()
			current_reward = 0

			terminal = False
			while not terminal:	
				action = self.select_action(self.state)
				next_frame, reward, terminal = self.env.act(action)
				self.process_state(next_frame)

				current_reward += reward
				if terminal:
					break
				
			if current_reward > best_reward:
				best_reward = current_reward
			print("<%d> Current reward: %d" % (episode, current_reward))
		print("="*30)
		print("Best reward: %d" % (best_reward))

	def select_action(self, state):
		if self.args.train:
			self.eps = np.max([self.args.eps_min, self.args.eps_init - (self.args.eps_init - self.args.eps_min)*(float(self.step)/float(self.args.final_exploration_frame))])
		else:
			self.eps = self.args.eps_test

		if np.random.rand() < self.eps:
			action = self.env.random_action()
			self.random_histogram[action] += 1
		else:
			#action = np.argmax(self.dqn.predict_Q(state))
			q = self.dqn.predict_Q_value(state)[0]
			action_candidate = np.argwhere(q == np.max(q))
			if len(action_candidate) > 1:
				action = action_candidate[np.random.randint(0,len(action_candidate))][0]
			else:
				action = action_candidate[0][0]
			self.action_histogram[action] += 1
		return action

	@property
	def model_dir(self):
		if self.args.load_env_name == None:
			return '{}_{}_batch'.format(self.args.env_name, self.args.batch_size)
		else:
			return '{}_{}_batch'.format(self.args.load_env_name, self.args.batch_size)

	def save(self):
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, str(self.step)))
		print('*** Save at %d steps' % self.step)

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

class SimpleAgent(Agent):
	def __init__(self, args, sess):
		self.env = SimpleEnvironment(args)
		self.memory = SimpleMemory(args, self.env.frame_shape)
		super(SimpleAgent, self).__init__(args, sess)

	def reset_episode(self):
		self.state = self.env.new_episode()

	def process_state(self, next_frame):
		self.state = next_frame

class AtariAgent(Agent):
	def __init__(self, args, sess):
		self.env = AtariEnvironment(args)
		self.memory = AtariMemory(args, self.env.frame_shape)
		super(AtariAgent, self).__init__(args, sess)

	def reset_episode(self):
		self.state = np.zeros(self.memory.state_shape, dtype=np.float32)
		self.state[:,:,self.args.state_length-1] = self.env.new_episode()

	def process_state(self, next_frame):
		self.state[:, :, 0:self.args.state_length-1] = self.state[:, :, 1:self.args.state_length]
		self.state[:, :, self.args.state_length-1] = next_frame