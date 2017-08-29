import cv2
import gym
import random
import numpy as np
from time import sleep

def rgb2gray(image):
	return np.dot(image[:,:,0:3], [0.2990, 0.5870, 0.1140])

def process_frame(image, size):
	#return cv2.resize(rgb2gray(image)/255, size)
	img = cv2.resize(rgb2gray(image), (84, 110)) # parameter of cv2.resize : width x height (# of cols, # of rows)
	return img[26:110, :] # In breakout-v0, this region show just game board not including the score board.
	
class Environment():
	def __init__(self, args):
		self.args = args

		self.env = gym.make(self.args.env_name)
		self.env.reset()

		self.num_actions = self.env.action_space.n
		self.input_size = list(self.env.observation_space.shape)
		self.frame_size = (self.args.frame_height, self.args.frame_width)
	
	def random_action(self):
		return self.env.action_space.sample()
	
	def _render(self):
		if self.args.display:
			self.env.render()
			sleep(self.args.display_interval)

class SimpleEnvironment(Environment):
	def new_episode(self):
		self.state = self.env.reset()
		return self.state
	
	def act(self, action):
		self.state, self.reward, self.terminal, _ = self.env.step(action)
		self._render()
		return self.state, self.reward, self.terminal


class AtariEnvironment(Environment):
	def new_episode(self):
		#if self.env.unwrapped.ale.lives() == 0:
		#	self.env.reset()
		self.env.reset()
		#self.frame, _, _, _ = self.env.step(0) 
		#self.frame, _, _, _ = self.env.step(self.env.action_space.sample())
		self.frame, _, _, _ = self.env.step(1)
		self._render()
		state = np.zeros([self.args.state_length, self.args.frame_height, self.args.frame_width], dtype=np.float32)
		self.frame = process_frame(self.frame, self.frame_size)
		for i in xrange(4):
			state[i] = self.frame
		self.lives = self.env.unwrapped.ale.lives()
		return state

	def new_episode_random_start(self):
		self.new_game()
		for _ in xrange(random.randint(0, self.args.random_start - 1)):
			#self.env.step(0)
			self.frame, _, _, _ = self.env.step(self.env.action_space.sample())
		self._render()
		state = np.zeros([self.args.state_length, self.args.frame_height, self.args.frame_width], dtype=np.float32)
		self.frame = process_frame(self.frame, self.frame_size)
		for i in xrange(4):
			state[i] = self.frame
		return state

	def act(self, action):
		for _ in xrange(self.args.num_skipping_frames):
			self.frame, self.reward, self.terminal, _ = self.env.step(action)
			
			current_lives = self.env.unwrapped.ale.lives()
			if self.args.train and self.lives>current_lives:
				self.terminal = True
			#if self.terminal:
				self.reward = -1
				break

		self._render()
		return process_frame(self.frame, self.frame_size), self.reward, self.terminal
