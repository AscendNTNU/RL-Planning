import subprocess
from time import time
import numpy as np
import tensorflow as tf

from util import * 
from sim_variable_setup import *
import network
tf.reset_default_graph()

# This must match the network that was trained
h_size = 128
learning_rate = 1e-4        
targetQN = network.Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size)
mainQN = network.Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size)

saver = tf.train.Saver()


with tf.Session() as sess:
	last_time = 0
	path = "./dqn"             #The path to save our model to.
	print('Loading Model...')
	ckpt = tf.train.get_checkpoint_state(path)
	saver.restore(sess,ckpt.model_checkpoint_path)

	sim.initialize_gui()
	process = subprocess.Popen("../build/sim")
	step = sim.recieve_state_gui()
	while step.ai_data_input.elapsed_time - last_time < 5:
	    step = sim.recieve_state_gui()

	state = observation_to_input_array(step.ai_data_input)
	done = False
	total_reward = 0
	#The Q-Network
	while True: #If the agent takes longer than max time, end the trial.
	    action = sess.run(targetQN.predict,feed_dict={targetQN.inputs:[state], targetQN.dropout_ratio:1})
	    action = action[0]
	    reward = 0 #sim.action_rewards(action_pool[action])/1000;
	    print(action_pool[action])
	    sim.send_command_gui(action_pool[action])
	    step = sim.recieve_state_gui()
	    while step.ai_data_input.elapsed_time - last_time < 5 or not step.cmd_done:
	        reward += step.reward
	        step = sim.recieve_state_gui()
	    last_time = step.ai_data_input.elapsed_time
	    print("cmd_done")

	    state = observation_to_input_array(step.ai_data_input);
	    
	    reward += step.reward
	    total_reward += reward

	    if done == True:
	        print("done")
	        break

	print("killing")
	process.kill()
	print(total_reward)
