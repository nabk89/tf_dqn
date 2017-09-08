import cv2
import gym
import numpy as np
from time import sleep

class Environment(object):
	def __init__(self, args):
		self.args = args
		self.env = gym.make(self.args.env_name)
		self.num_actions = self.env.action_space.n
	
	def random_action(self):
		return self.env.action_space.sample()
	
	def _render(self):
		if self.args.display:
			self.env.render()
			sleep(self.args.display_interval)


class SimpleEnvironment(Environment):
	def __init(self, args):
		super(SimpleEnvironment, self).__init__(args)
		self.frame_shape = list(self.env.observation_space.shape)

	def new_episode(self):
		return self.env.reset()
	
	def act(self, action):
		self.state, self.reward, self.terminal, _ = self.env.step(action)
		'''
		if self.terminal:
			self.reward = -1
		else:
			self.reward = 0.1
		'''
		self._render()
		return self.state, self.reward, self.terminal


class AtariEnvironment(Environment):
	def __init__(self, args):
		super(AtariEnvironment, self).__init__(args)
		self.frame_shape = [self.args.frame_height, self.args.frame_width]

	def new_episode(self):
		'''
		if self.env.unwrapped.ale.lives() == 0:
			self.env.reset()
		'''
		self.frame = self.env.reset()
		self._render()
		#self.lives = self.env.unwrapped.ale.lives()
		return self.process_frame()

	def act(self, action):
		total_reward = 0
		for _ in xrange(self.args.num_skipping_frames):
			self.frame, self.reward, self.terminal, _ = self.env.step(action)
			total_reward += self.reward
			'''
			# Uncomment when training with just one life ((ex) Breakout)
			current_lives = self.env.unwrapped.ale.lives()
			if self.args.train and self.lives>current_lives:
				self.terminal = True
				self.lives = current_lives
			'''
			if self.terminal:
				break
		if total_reward == 0:
			self.reward = 0.0
		elif total_reward > 0: # positive rewards = 1
			self.reward = 1.0
		else: # negative rewards = -1
			self.reward = -1.0

		self._render()
		return self.process_frame(), self.reward, self.terminal

	def process_frame(self):
		self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		# parameter of cv2.resize : width x height (# of cols, # of rows)
		self.frame = cv2.resize(self.frame, (84, 110)) 
		return self.frame[26:110, :]/255.0 # This region show just game board not including the score board.
