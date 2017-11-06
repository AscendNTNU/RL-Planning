import subprocess
from time import time
import numpy as np
import tensorflow as tf

from util import * 
from sim_variable_setup import * 

action_pool = list(range(0,3))

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,0.001)

saver = tf.train.Saver()


with tf.Session() as sess:
	last_time = 0
	path = "./dqn"             #The path to save our model to.
	print('Loading Model...')
	ckpt = tf.train.get_checkpoint_state(path)
	saver.restore(sess,ckpt.model_checkpoint_path)
	updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

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
	    action = sess.run(targetQN.predict,feed_dict={targetQN.inputs:[state], targetQN.keep_per:1})
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
