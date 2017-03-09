import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
import subprocess
from ctypes import *
from time import time

_sim = CDLL('./PythonAccessToSim.so')

Num_Targets =_sim.get_sim_Num_Targets()
D = 3+3*Num_Targets # input dimensionality

class ai_data_input_struct(Structure):
	_fields_ = [("elapsed_time", c_float),
				("drone_x", c_float),
				("drone_y", c_float),
				("target_x", c_float*Num_Targets),
				("target_y", c_float*Num_Targets),
				("target_q", c_float*Num_Targets)]

class step_result(Structure):
	_fields_ = [("ai_data_input", ai_data_input_struct),
				("reward", c_float),
				("done", c_bool),
				("cmd_done", c_bool)]

_sim.step.restype = step_result
_sim.send_command.restype = c_int
_sim.initialize.restype = c_int
_sim.recieve_state_gui.restype = step_result
_sim.send_command_gui.restype = c_int
_sim.action_rewards.restype = c_int

class Qnetwork():
	def __init__(self, lr, s_size,a_size,h_size, batch_size, myScope):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		initializer = tf.contrib.layers.xavier_initializer()
		self.inputs= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
		self.hidden = slim.fully_connected(self.inputs,h_size,biases_initializer=None,weights_initializer = initializer, activation_fn=tf.nn.relu)
		self.hidden2 = slim.fully_connected(self.hidden,h_size,biases_initializer=None,weights_initializer = initializer, activation_fn=tf.nn.relu)
		self.hidden3 = slim.fully_connected(self.hidden2,h_size,biases_initializer=None,weights_initializer = initializer, activation_fn=tf.nn.relu)
		self.streamAC,self.streamVC = tf.split(1,2,self.hidden3)
		self.streamA = tf.contrib.layers.flatten(self.streamAC)
		self.streamV = tf.contrib.layers.flatten(self.streamVC)
		self.AW = tf.Variable(tf.random_normal([int(h_size/2), a_size]))
		self.VW = tf.Variable(tf.random_normal([int(h_size/2), 1]))
		self.Advantage = tf.matmul(self.streamA,self.AW)
		self.Value = tf.matmul(self.streamV,self.VW)
		
		#Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.sub(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001) ## replace with variable TODO
		self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
	def __init__(self, buffer_size = 1000000):
		self.buffer = []
		self.buffer_size = buffer_size
	
	def add(self,experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
		self.buffer.extend(experience)
			
	def sample(self,size):
		return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


def resetGradBuffer(gradBuffer):
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
	return gradBuffer
		
def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:int(total_vars/2)]):
		
		op_holder.append(tfVars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tfVars[idx+int(total_vars/2)].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

def observation_to_input_array(ai_view):
	#print(ai_view.elapsed_time)
	result = np.array([ai_view.elapsed_time%20, ai_view.drone_x/20.0, ai_view.drone_y/20.0])
	for i in range(Num_Targets):
		result = np.append(result, ai_view.target_x[i]/20.0)
		result = np.append(result, ai_view.target_y[i]/20.0)
		result = np.append(result, ai_view.target_q[i]/(3.14159265359*2))
	return result


tf.reset_default_graph() #Clear the Tensorflow graph.



batch_size = 50 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .995 #Discount factor on the target Q-values
startE = 1#Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 6000000 #How many steps of training to reduce startE to endE.
num_episodes = 10000000#How many episodes of game environment to train network with.
pre_train_steps = 100000 #How many steps of random actions before training begins.
#max_epLength = 6*10 #The max allowed length of our episode.
load_model = False#Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 256#Hidden layer size
tau = 0.001 #Rate to update target network toward primary network
learning_rate = 1e-4
action_pool = list(range(0,3*Num_Targets))

render = False
tf.reset_default_graph()

#cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size,state_is_tuple=True)
mainQN = Qnetwork(lr=learning_rate,s_size=D,a_size=len(action_pool),h_size=h_size, batch_size = batch_size, myScope = "main")
targetQN = Qnetwork(lr=learning_rate,s_size=D,a_size=len(action_pool),h_size=h_size, batch_size = batch_size,  myScope = "target") #Load the agent.


init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = [0]
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
	os.makedirs(path)

with tf.Session() as sess:

	if load_model == True:
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		sess.run(init)
	summary_writer = tf.summary.FileWriter("dqn_summary")
	
	updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
	if render:
		_sim.initialize_gui()

	for i in range(num_episodes):
		episodeBuffer = experience_buffer()
		#Reset environment and get first new observation
		last_time = 0
		if render:
			#TODO:COULD CAUSE PROBLEMS
			process = subprocess.Popen("build/sim")
			step = _sim.recieve_state_gui()
			while step.ai_data_input.elapsed_time - last_time < 2:
				step = _sim.recieve_state_gui()
			last_time = step.ai_data_input.elapsed_time
		else:
			_sim.initialize()
			step = _sim.step()

			while(step.ai_data_input.drone_x < 0):
				sim.initialize()
				step = _sim.step()

		s = observation_to_input_array(step.ai_data_input)
		d = False
		xList = []
		rAll = 0
		j = 0
		#The Q-Network
		while True: #If the agent takes longer than max time, end the trial.
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			if np.random.rand(1) < e and not render:# or total_steps < pre_train_steps:
				a = np.random.randint(0,3*Num_Targets)
			else:
				a = sess.run(mainQN.predict,feed_dict={mainQN.inputs:[s]})[0]
				#actionDist = sess.run(mainQN.Qout,feed_dict={mainQN.inputs:[s]})[0]
				#print(actionDist)
			r =_sim.action_rewards(action_pool[a])/1000;

			if render:
				_sim.send_command_gui(action_pool[a])
				step = _sim.recieve_state_gui()
				while step.ai_data_input.elapsed_time - last_time < 2 or not step.cmd_done:
					r += step.reward
					step = _sim.recieve_state_gui()
				last_time = step.ai_data_input.elapsed_time
			else:
				_sim.send_command(action_pool[a])
				step = _sim.step()
				while not step.cmd_done:
					r += step.reward
					step = _sim.step()
				

			s1 = observation_to_input_array(step.ai_data_input);
			
			r += step.reward
			d = step.done
			total_steps += 1
			episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
			#print(a,r, step.ai_data_input.elapsed_time, step.ai_data_input.drone_x,step.ai_data_input.drone_y,step.ai_data_input.target_x[0],step.ai_data_input.target_y[0], step.ai_data_input.target_q[0])

			if total_steps > pre_train_steps and not render:
				if e > endE:
					e -= stepDrop
				
				if total_steps % (update_freq) == 0:
					trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
					#Below we perform the Double-DQN update to the target Q-values
					Q1 = sess.run(mainQN.predict,feed_dict={mainQN.inputs:np.vstack(trainBatch[:,3])})
					Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.inputs:np.vstack(trainBatch[:,3])})
					end_multiplier = -(trainBatch[:,4] - 1)
					doubleQ = Q2[range(batch_size),Q1]
					targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
					#Update the network with our target values.
					_ = sess.run(mainQN.updateModel,
						feed_dict={mainQN.inputs:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
					
					updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
			rAll += r
			s = s1

			if d == True:
				break

		
		if render:
			print("killing")
			process.kill()
			print(rAll)

		#Get all experiences from this episode and discount their rewards.
		myBuffer.add(episodeBuffer.buffer)
		jList.append(j)
		rList.append(rAll)
		#print(rAll)
		#Periodically save the model. 
		if i % 50 == 0 and i != 0 and not render:

			mean_reward = np.mean(rList[-50:])
			mean_length = np.mean(jList[-50:])
			summary = tf.Summary()
			summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
			summary.value.add(tag='Perf/Length', simple_value=float(mean_length))

			summary_writer.add_summary(summary, i)
			print(i, "Mean reward: ", mean_reward, " Mean steps: ", mean_length, " e: ", e)

			summary_writer.flush( 	)

		if i % 1000 == 0 and i != 0 and not render:
			saver.save(sess,path+'/model-'+str(i)+'.cptk')
			print("Saved Model")



	saver.save(sess,path+'/model-'+str(i)+'.cptk')

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
