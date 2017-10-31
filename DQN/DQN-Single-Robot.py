import os
import subprocess

from ctypes import *
from time import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from DQN_Helper import * 


_sim = CDLL('../cppWrapper/PythonAccessToSim.so')

Num_Targets =_sim.get_sim_Num_Targets()
D = 3+4 # input dimensionality

class ai_data_input_struct(Structure):
    _fields_ = [("elapsed_time", c_float),
                ("drone_x", c_float),
                ("drone_y", c_float),
                ("target_x", c_float*Num_Targets),
                ("target_y", c_float*Num_Targets),
                ("target_q", c_float*Num_Targets),
                ("target_removed", c_int*Num_Targets)]

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

# Neural network setup
class Qnetwork():
    def __init__(self, lr, s_size,a_size, h_size, batch_size, scope):
        
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.inputs= tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=(), dtype=tf.float32)
        self.hidden = layers.fully_connected(self.inputs, h_size)
        self.hidden = layers.dropout(self.hidden, self.keep_per)

        self.streamAC, self.streamVC = tf.split(self.hidden, 2, 1)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([int(h_size/2), a_size]))
        self.VW = tf.Variable(tf.random_normal([int(h_size/2), 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        self.Q_dist = layers.softmax(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

batch_size = 50             #How many experiences to use for each training step.
update_freq = 4             #How often to perform a training step.
y = .995                    #Discount factor on the target Q-values
startE = 0.1                #Starting chance of random action
endE = 0.1                  #Final chance of random action
anneling_steps = 100000    #How many steps of training to reduce startE to endE.
num_episodes = 10000     #How many episodes of game environment to train network with.
pre_train_steps = 10000   #How many     steps of random actions before training begins.
path = "./dqn1"             #The path to save our model to.
h_size = 256                #Hidden layer size
tau = 0.001                 #Rate to update target network toward primary network

learning_rate = 1e-4        
action_pool = list(range(0,3))

training = True          #Whether to train or not.
load_model = False  


tf.reset_default_graph()

mainQN = Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size, batch_size=batch_size, scope="main")
targetQN = Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size, batch_size=batch_size,  scope="target") #Load the agent.
init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)
experience_buffer = ExperienceBuffer(buffer_size=pre_train_steps)

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
steps_per_episode = []
reward_per_episode = [0]
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if training:
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter("dqn1_summary")
        
        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
        for i in range(num_episodes):

            #Reset Variables
            last_time = 0
            done = False
            xList = []
            total_reward = 0
            steps = 0

            #Reset environment
            _sim.initialize()
            step = _sim.step()
            while(step.ai_data_input.drone_x < 0):
                _sim.initialize()
                step = _sim.step()

            state = observation_to_input_array(step.ai_data_input)

            while True:
                #Run one step
                steps += 1
                action = sess.run(mainQN.predict, feed_dict={mainQN.inputs:[state], mainQN.keep_per:(1-e)+0.1})
                action = action[0]
                reward = 0

                _sim.send_command(action_pool[action])
                step = _sim.step()
                while not step.cmd_done:
                    reward += step.reward
                    step = _sim.step()
                    
                next_state = observation_to_input_array(step.ai_data_input);

                reward += step.reward
                done = step.done
                total_steps += 1
                experience_buffer.add(Experience(state, action, reward, next_state, done))
                #Save the experience to our episode buffer.

                # If we are done with experience collection start updating Q network
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % update_freq == 0:

                        states_batch, action_batch, reward_batch, next_states_batch, done_batch = experience_buffer.sample(batch_size) #Get action random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.inputs:next_states_batch, mainQN.keep_per:1.0}) #TODO: if this breaks try np.vstack
                        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.inputs:next_states_batch, targetQN.keep_per:1.0})
                        end_multiplier = -(done_batch - 1)
                        doubleQ = Q2[range(batch_size),Q1]
                        targetQ = reward_batch + (y*doubleQ * end_multiplier)
                        
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateModel,
                            feed_dict={mainQN.inputs:states_batch,
                                       mainQN.targetQ:targetQ,
                                       mainQN.actions:action_batch,
                                       mainQN.keep_per:1.0})
                        
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                
                total_reward += reward
                state = next_state

                if done:
                    break

            #Get all experiences from this episode and discount their rewards.
            steps_per_episode.append(steps)
            reward_per_episode.append(total_reward)

            #Periodically save the model. 
            if i % 50 == 0 and i != 0:

                mean_reward = np.mean(reward_per_episode[-50:])
                mean_length = np.mean(steps_per_episode[-50:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))

                summary_writer.add_summary(summary, i)
                print(i, "Mean reward: ", mean_reward, " Mean steps: ", mean_length, " e: ", e)

                summary_writer.flush(   )

            if i % 1000 == 0 and i != 0:
                saver.save(sess,path+'/model-'+str(i)+'.cptk')
                print("Saved Model")
        
    else:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.

        _sim.initialize_gui()

        for i in range(num_episodes):
            #Reset environment and get first new observation
            last_time = 0
            process = subprocess.Popen("build/sim")
            step = _sim.recieve_state_gui()
            while step.ai_data_input.elapsed_time - last_time < 5:
                step = _sim.recieve_state_gui()

            target = chooseRobot(step.ai_data_input)
            state = observation_to_input_array(step.ai_data_input,target)
            done = False
            total_reward = 0
            #The Q-Network
            while True: #If the agent takes longer than max time, end the trial.
                action = sess.run(targetQN.predict,feed_dict={targetQN.inputs:state, targetQN.keep_per:1})
                action = action[0]
                reward = 0 #_sim.action_rewards(action_pool[action])/1000;

                _sim.target_send_command_gui(action_pool[action], target)
                step = _sim.recieve_state_gui()
                while step.ai_data_input.elapsed_time - last_time < 5 or not step.cmd_done:
                    reward += step.reward
                    step = _sim.recieve_state_gui()
                    print("stepping")
                    print(step.ai_data_input.target_y[target])
                last_time = step.ai_data_input.elapsed_time
                print("cmd_done")
            
                if step.ai_data_input.target_removed[target]:
                    print("choosing new robot")
                    print(step.ai_data_input.target_y[target])  
                    target = chooseRobot(step.ai_data_input)
                next_state = target_observation_to_input_array(step.ai_data_input,target);
                
                reward += step.reward
                total_reward += reward

                if done == True:
                    print("done")
                    break
            
            print("killing")
            process.kill()
            print(total_reward)
