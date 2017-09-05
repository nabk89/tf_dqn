import cv2
import gym
import numpy as np
import tensorflow as tf

import readchar

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5
#dict_keys = {'\x1b[A': FIRE, '\x1b[C': RIGHT, 'x1b[D': LEFT}
dict_keys = {'k': NOOP, 'l': FIRE, '.': RIGHT, ',': LEFT, 'b': LEFTFIRE, 'n': RIGHTFIRE}

#env = gym.make('Breakout-v0')
env = gym.make('Pong-v0')
env.reset()
env.render()

print env.action_space.n
print env.unwrapped.get_action_meanings()

total_reward = 0
while True:
	key = readchar.readkey()
	print key + " " + str(dict_keys[key])
	if key not in dict_keys.keys():
		print("Not allowed keys!")
		break

	action = dict_keys[key]
	state, reward, terminal, _ = env.step(action)
	print str(reward) + " " + str(terminal)
	total_reward += reward
	env.render()

	if terminal:
		print("Finish (reward: %d)" % total_reward)
		break
