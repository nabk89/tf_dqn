import random, os, time
import numpy as np

class SimpleMemory:
	def __init__(self, args, environment):
		self.args = args
		self.memory_size = self.args.memory_size
		self.env = environment
		self.count = 0
		self.current = 0
		
		self.states = np.empty([self.memory_size] + self.env.input_size, dtype=np.float16)
		self.actions = np.empty(self.memory_size, dtype=np.uint8)
		self.rewards = np.empty(self.memory_size, dtype=np.integer)
		self.terminals = np.empty(self.memory_size, dtype=np.bool)

		self.prestates = np.empty([self.args.batch_size] + self.env.input_size, dtype=np.float16)
		self.poststates = np.empty([self.args.batch_size] + self.env.input_size, dtype=np.float16)

	def add(self, action, reward, terminal, next_state):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.terminals[self.current] = terminal
		self.states[self.current] = next_state
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.memory_size

	def mini_batch(self):
		batch_indices = []
		while len(batch_indices) < self.args.batch_size:
			while True:
				idx = random.randint(self.args.state_length-1, self.count-1)
				if idx == self.current+1:
					continue
				if self.terminals[idx-1]: #or self.terminals[idx]:
					continue
				break
			self.prestates[len(batch_indices)] = self.states[idx-1]
			self.poststates[len(batch_indices)] = self.states[idx]
			batch_indices.append(idx)
		actions = self.actions[batch_indices]
		rewards = self.rewards[batch_indices]
		terminals = self.rewards[batch_indices]
		return self.prestates, actions, rewards, terminals, self.poststates


class ReplayMemory:
	def __init__(self, args):
		self.args = args
		self.memory_size = self.args.memory_size
		self.count = 0
		self.current = 0
		
		self.frames = np.empty((self.memory_size, args.frame_height, args.frame_width), dtype=np.float16)
		self.actions = np.empty(self.memory_size, dtype=np.uint8)
		self.rewards = np.empty(self.memory_size, dtype=np.integer)
		self.terminals = np.empty(self.memory_size, dtype=np.bool)

		self.current_states = np.empty((self.args.batch_size, self.args.state_length, self.args.frame_height, self.args.frame_width), dtype=np.float16)
		self.next_states = np.empty((self.args.batch_size, self.args.state_length, self.args.frame_height, self.args.frame_width), dtype=np.float16)

	def add(self, action, reward, terminal, next_frame):
		self.frames[self.current] = next_frame
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current+1)
		self.current = (self.current + 1) % self.memory_size

	def _make_state(self, frame_idx):
		state = self.frames[(frame_idx - self.args.state_length + 1):(frame_idx+1)]

	def mini_batch(self):
		batch_indices = []
		while len(batch_indices) < self.args.batch_size:
			while True:
				frame_idx = random.randint(self.args.state_length-1, self.count-1)
				if frame_idx >= self.current and frame_idx-self.args.state_length < self.current:
					continue
				if self.terminals[(frame_idx - self.args.state_length):frame_idx-1].any():
					continue
				break
			self.current_states[len(batch_indices)] = self._make_state(frame_idx-1)
			self.next_states[len(batch_indices)] = self._make_state(frame_idx)
			batch_indices.append(frame_idx)

		actions = self.actions[batch_indices]
		rewards = self.rewards[batch_indices]
		terminals = self.terminals[batch_indices]

		return self.current_states, actions, rewards, terminals, self.next_states
			
