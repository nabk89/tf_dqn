import random
import tensorflow as tf

from agent import *

import argparse, time, os, sys

def str2bool(s):
	if s.lower() in ('yes', 'y', '1', 'true', 't'):
		return True
	elif s.lower() in ('no', 'n', '0', 'false', 'f'):
		return False

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str2bool, default='true')
	parser.add_argument('--display', type=str2bool, default='false')
	parser.add_argument('--display_interval', type=float, default=0.2)
	parser.add_argument('--nips', type=str2bool, default='true')
	parser.add_argument('--max_step', type=int, default=10000000)
	parser.add_argument('--final_exploration_frame', type=int, default=1000000)
	parser.add_argument('--memory_size', type=int, default=1000000)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_skipping_frames', type=int, default=4)
	parser.add_argument('--state_length', type=int, default=4)
	parser.add_argument('--frame_width', type=int, default=84)
	parser.add_argument('--frame_height', type=int, default=84)
	parser.add_argument('--discount_factor', type=float, default=0.99)
	parser.add_argument('--max_reward', type=int, default=1)
	parser.add_argument('--min_reward', type=int, default=-1)
	parser.add_argument('--eps_init', type=float, default=1.0)
	parser.add_argument('--eps_min', type=float, default=0.1)
	parser.add_argument('--eps_test', type=float, default=0.05)
	#parser.add_argument('--eps_step', type=float, default=1e7)
	parser.add_argument('--learning_rate', type=float, default=0.00025)
	parser.add_argument('--random_start', type=int, default=30)
	parser.add_argument('--training_start_step', type=int, default=5000)
	parser.add_argument('--train_freq', type=int, default=4)
	parser.add_argument('--copy_freq', type=int, default=10000)
	parser.add_argument('--save_freq', type=int, default=50000)
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
	parser.add_argument('--log_dir', type=str, default='./log')
	parser.add_argument('--env_name', type=str, default='Breakout-v0')
	parser.add_argument('--simple', type=str2bool, default='false')
	parser.add_argument('--show_freq', type=int, default=2000)

	myArgs = parser.parse_args()
	if not os.path.exists(myArgs.checkpoint_dir):
		os.makedirs(myArgs.checkpoint_dir)
	if not os.path.exists(myArgs.log_dir):
		os.makedirs(myArgs.log_dir)

	if myArgs.simple:
		makeSimpleArgs(myArgs)

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	run_config = tf.ConfigProto()
	run_config.log_device_placement = False
	run_config.gpu_options.allow_growth = True

	with tf.Session(config = run_config) as sess:
		myAgent = Agent(myArgs, sess)
		#sess.run(tf.global_variables_initializer())
		if myArgs.train:
			myAgent.train()
		else:
			myAgent.play()

def makeSimpleArgs(args):
	args.env_name = 'CartPole-v0'
	args.memory_size = 10000
	args.max_step = 100000
	args.save_freq = 1000
	args.discount_factor = 0.95
	args.eps_init = 1
	args.eps_min = 0.1
	args.train_freq = 1 
	args.copy_freq = 5000
	args.training_start_step = 100
	args.learning_rate = 0.001
	args.final_exploration_frame = 10000
	args.display_interval = 0.05


if __name__ == '__main__':
	main()
