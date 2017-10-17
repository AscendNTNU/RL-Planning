import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ctypes import *
from helper import *

from random import choice
from time import time
import os

import subprocess
import time
import string

_numTargets = CDLL('./PythonAccessToSim.so')

Num_Targets =_numTargets.get_sim_Num_Targets()
D = 3+3*Num_Targets # input dimensionality
startE = 1
endE = 0.1

anneling_steps = 100000
stepDrop = (startE - endE)/anneling_steps

class ai_data_input_struct(Structure):
	_fields_ = [("elapsed_time", c_float),
				("drone_x", c_int),
				("drone_y", c_int),
				("target_x", c_int*Num_Targets),
				("target_y", c_int*Num_Targets),
				("target_q", c_float*Num_Targets)]

class step_result(Structure):
	_fields_ = [("ai_data_input", ai_data_input_struct),
				("reward", c_float),
				("done", c_bool),
				("cmd_done", c_bool)]

#Ctypes assumes return type is int


def observation_to_input_array(ai_view):
	#print(ai_view.elapsed_time)
	result = np.array([ai_view.elapsed_time%20/20, ai_view.drone_x/20, ai_view.drone_y/20])
	for i in range(Num_Targets):
		result = np.append(result, ai_view.target_x[i]/20)
		result = np.append(result, ai_view.target_y[i]/20)
		result = np.append(result, ai_view.target_q[i]/(3.14159265359*2))
	return result


class AC_Network():
	def __init__(self,s_size,a_size,scope,trainer):
		with tf.variable_scope(scope):
			initializer = tf.contrib.layers.xavier_initializer()

			self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)

			hidden = slim.fully_connected(self.inputs,256,biases_initializer=None,activation_fn=tf.nn.elu)

			#Input and visual encoding layers
			# self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
			# self.imageIn = tf.reshape(self.inputs,shape=[-1,5,1])
			# self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
			# 	inputs=self.imageIn,num_outputs=16,
			# 	kernel_size=[3],stride=[2],padding='VALID')
			# self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
			# 	inputs=self.conv1,num_outputs=32,
			# 	kernel_size=[2],stride=[1],padding='VALID')
			# hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
			
			#Recurrent network for temporal dependencies
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256,state_is_tuple=True)
			c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
			h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
			self.state_init = [c_init, h_init]
			c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
			h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
			self.state_in = (c_in, h_in)
			rnn_in = tf.expand_dims(hidden, [0])
			step_size = tf.shape(self.inputs)[:1]
			state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
				lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
				time_major=False)
			lstm_c, lstm_h = lstm_state
			self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
			rnn_out = tf.reshape(lstm_outputs, [-1, 256])
			
			#Output layers for policy and value estimations
			self.policy = slim.fully_connected(rnn_out,a_size,
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(0.01),
				biases_initializer=None)
			self.value = slim.fully_connected(rnn_out,1,
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(1.0),
				biases_initializer=None)
			
			#Only the worker network need ops for loss functions and gradient updating.
			if scope != 'global':
				self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
				self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

				#Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
				self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy+10e-6))
				self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
				self.loss = 0.5 * self.value_loss + self.policy_loss# - self.entropy * 0.01

				#Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,1.0)
				
				#Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
	def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes):
		self.name = "worker_" + str(name)
		self.number = name        
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
		self.sleep_time = 1
		#Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
		self.update_local_ops = update_target_graph('global',self.name)        
		
		#The Below code is related to setting up the Doom environment
		
		#Todo:
		self.actions = list(range(0,3*Num_Targets))
		# for i in range(Num_Targets):
		# 	self.actions.append(2*i)
		# 	self.actions.append(2*i+1)
		self.env = CDLL('./PythonAccessToSim.so')
		self.env.step.restype = step_result
		self.env.send_command.restype = c_int
		self.env.initialize.restype = c_int
		self.env.recieve_state_gui.restype = step_result
		self.last_time = 0
		self.e = 1

		
	def train(self,rollout,sess,gamma,bootstrap_value):
		#TODO:
		rollout = np.array(rollout)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]
		
		# Here we take the rewards and values from the rollout, and use them to 
		# generate the advantage and discounted returns. 
		# The advantage function uses "Generalized Advantage Estimation"
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		#rnn_state = self.local_AC.state_init
		rnn_state = self.local_AC.state_init
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs:np.vstack(observations),
			self.local_AC.actions:actions,
			self.local_AC.advantages:advantages,
			self.local_AC.state_in[0]:rnn_state[0],
			self.local_AC.state_in[1]:rnn_state[1]}
		v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.apply_grads],
			feed_dict=feed_dict)
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
		
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print("Starting worker " + str(self.number))
		if render:
			self.env.initialize_gui()
		with sess.as_default(), sess.graph.as_default():
			running_reward = 0
			while not coord.should_stop():
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_values = []
				episode_reward = 0
				episode_step_count = 0
				d = False
				
				if render:

					process = subprocess.Popen("build/sim")
					self.step = self.env.recieve_state_gui()
					s = observation_to_input_array(self.step.ai_data_input);
				else:
					self.env.initialize()
					self.step = self.env.step()
					s = observation_to_input_array(self.step.ai_data_input);

				rnn_state = self.local_AC.state_init
				while True:
					#Take an action using probabilities from policy network output.
					#print(self.step.ai_data_input.elapsed_time - self.last_time)
					#and self.step.ai_data_input.elapsed_time - self.last_time > 5.0:
					r = 0
					a_dist,v,rnn_state  = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
						feed_dict={self.local_AC.inputs:[s],
						self.local_AC.state_in[0]:rnn_state[0],
						self.local_AC.state_in[1]:rnn_state[1]})	
					# a = np.random.choice(a_dist[0],p=a_dist[0])
					# a = np.argmax(a_dist == a)
					#print(a_dist)

					if np.random.rand(1) < self.e and not render:# or total_steps < pre_train_steps:
						a = np.random.randint(0,3*Num_Targets)
					else:
						a = np.argmax(a_dist == a)

					if render:
						self.env.send_command_gui(self.actions[a])	
						self.step = self.env.recieve_state_gui()
						while self.step.cmd_done == 0:
							r += self.step.reward
							self.step = self.env.recieve_state_gui()
					else:
						self.env.send_command(self.actions[a])
						self.step = self.env.step()
						while self.step.cmd_done == 0:
							r += self.step.reward
							self.step = self.env.step()
					d = self.step.done
					r += self.step.reward
					if d == False:
						s1 = observation_to_input_array(self.step.ai_data_input);
					else:
						s1 = s
					episode_buffer.append([s,a,r,s1,d,v[0,0]])
					episode_values.append(v[0,0])	

					episode_reward += r
					s = s1                    
					total_steps += 1
					episode_step_count += 1

					# If the episode hasn't ended, but the experience buffer is full, then we
					# make an update step using that experience rollout.
					if len(episode_buffer) == 1000 and d != True and episode_step_count != max_episode_length - 1:
						# Since we don't know what the true final return is, we "bootstrap" from our current
						# value estimation.
						v1 = sess.run(self.local_AC.value, 	
							feed_dict={self.local_AC.inputs:[s]})[0,0]
						v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if d == True:
						break

				#print("episode number %f done, reward: %f" % (episode_count, episode_reward))							
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				if self.e>0.1:
					self.e -= 0.0001
				# Update the networkk using the experience buffer at the end of the episode.
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
				# Periodically save gifsself.env.is_episode_finished() of episodes, model parameters, and summary statistics.
				if episode_count % 50 == 0 and episode_count != 0:
					if episode_count % 250 == 0 and self.name == 'worker_0':
						print("Saving Model")
						saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')

					mean_reward = np.mean(self.episode_rewards[-50:])
					mean_length = np.mean(self.episode_lengths[-50:])
					mean_value = np.mean(self.episode_mean_values[-50:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)
					print("Episode Number: ", episode_count, "Mean reward: ", mean_reward, "Last_action ", a)
					self.summary_writer.flush()
				if self.name == 'worker_0':
					sess.run(self.increment)
				episode_count += 1
				self.last_time = 0;
				if render:
					print("killing")
					process.kill()



max_episode_length = 200
gamma = .99# discount rate for advantage estimation and reward discounting
s_size = D #
a_size = Num_Targets*3# + 1# Agent can move, land or nothing
load_model = False
render = False	
model_path = './model'


tf.reset_default_graph()

if not os.path.exists(model_path):
	os.makedirs(model_path)
	
with tf.device("/cpu:0"): 
	global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
	trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99, epsilon=0.1)
	master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
	num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
	workers = []
	# Create worker classes
	for i in range(num_workers):
		workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes))
	saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	if load_model == True:
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())
		
	# This is where the asynchronous magic happens.
	# Start the "work" process for each worker in a separate thread.
	worker_threads = []
	for worker in workers:
		worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
		t = threading.Thread(target=(worker_work))
		t.start()
		worker_threads.append(t)
	print("Trying to join")
	coord.join(worker_threads)
	print("done woker joining")