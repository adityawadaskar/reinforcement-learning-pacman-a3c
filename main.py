import tensorflow as tf
import os
import sys
import threading
from network import A3C_Network
from agent import A3C_Worker
import argparse
import time

#Parse Inputs - Default parameters listed below
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, help='learning rate', default = 0.0001)
parser.add_argument('--gamma', type=float, help='discount factor for rewards', default = 0.88)
parser.add_argument('--workers', type=int, help='number of training processes', default = 16)
parser.add_argument('--num-steps', type=int, help='number of forward steps in A3C', default = 20)
parser.add_argument('--max-episode-length', type=int, help='maximum length of an episode', default = 6000)
parser.add_argument('--env',help='environment to train on', default = 'MsPacman-v0')
parser.add_argument('--a-size', type=int, help='number of actions available in MsPacman env', default = 9)
parser.add_argument('--render', help='render game', default = False)
parser.add_argument('--evaluate', help='if true, load and evaluate saved model', default = False)
parser.add_argument('--sleeptime', type=float, help='time between each test (seconds)', default = 15)
parser.add_argument('--skip-rate', type=int, help='frame skip rate', default = 4)
parser.add_argument('--load', help='load existing model', default = False)
parser.add_argument('--model-path', help='model path', default = './model')
args = parser.parse_args()
if args.evaluate:
	args.load = True

if __name__ == '__main__':
	tf.reset_default_graph()

	#Create directory to save model
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	with tf.device("/cpu:0"):
		global_network = A3C_Network('global', args.a_size, None) # Create global network
		trainer = tf.train.AdamOptimizer(learning_rate=args.lr) # Use Adam Optimizer for each thread
		workers = []
		args.start_time = time.time()

		# Create worker threads. Note: Last index contains test agent
		if not args.evaluate:
			for i in range(args.workers):
				workers.append(A3C_Worker(i, trainer, args))
		workers.append(A3C_Worker('test', trainer, args))

	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=3)
		if args.load:
			print ('Loading model...')
			ckpt = tf.train.get_checkpoint_state(args.model_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
			print ('Model loaded successfully.')
		else:
			sess.run(tf.global_variables_initializer())
			
		thread_manager = tf.train.Coordinator() #Coordinate asynchronous threads

		# Begin asynchronous training
		worker_threads = []
		
		if not args.evaluate: # If in training mode
			for i in range(args.workers):
				t = threading.Thread(target=workers[i].train, args=(sess, thread_manager))
				t.start()
				time.sleep(0.1)
				worker_threads.append(t)
		else:
			args.workers = 0	# If in evaluation mode
		
		t = threading.Thread(target=workers[args.workers].test, args=(sess, thread_manager, saver, args)) #Append test agent
		t.start()
		time.sleep(0.1)
		worker_threads.append(t)
		thread_manager.join(worker_threads)