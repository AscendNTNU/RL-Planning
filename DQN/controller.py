import tensorflow as tf
import numpy as np
from util import * 
from sim_variable_setup import *

class Controller():
    def __init__(self, batch_size, buffer_size, anneling_episodes, update_freq, gamma, trainables):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.experience_buffer = ExperienceBuffer(buffer_size=buffer_size, batch_size=batch_size)
        self.update_freq = update_freq
        self.gamma = gamma
        self.dropout_ratio = 0.1
        self.change_per_episode = (0.9-self.dropout_ratio)/anneling_episodes

        # Summary statistics
        self.reward_per_episode = []
        self.steps_per_episode = []
        self.episode_number = 1

        # Network updater
        self.targetOps = updateTargetGraph(trainables,0.001)


    def decreaseDropout(self):

        if self.dropout_ratio < 0.9:
            self.dropout_ratio += self.change_per_episode


    def runStep(self, state, network):
        '''
        Runs one step of the mission using a network to choose action.
        '''

        action = self.sess.run(
            network.predict,
            feed_dict={network.inputs:[state],
                       network.dropout_ratio:self.dropout_ratio})

        action = action[0]
        sim.send_command(action_pool[action])
        return action, sim.step() 

    def runEpisode(self, mainQN):
        '''
        Runs one episode of the mission
        '''
        total_reward = 0
        steps = 0

        sim.initialize()
        step = sim.step()
        state = observation_to_input_array(step.ai_data_input)

        while True:
            steps += 1

            action, step = self.runStep(state, mainQN)
            reward = step.reward
            total_reward += reward
            
            next_state = observation_to_input_array(step.ai_data_input)
            self.experience_buffer.add(Experience(state, action, reward, next_state, step.done))
            
            state = next_state
            if step.done:
                self.steps_per_episode.append(steps)
                self.reward_per_episode.append(total_reward)
                self.episode_number += 1
                return

    def trainNetwork(self, mainQN, targetQN):
        '''
        Trains the network
        '''

        #Replay a batch of random experiences
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = self.experience_buffer.sample
        
        #Below we perform the Double-DQN update to the target Q-values
        best_actions_index = self.sess.run(mainQN.predict,feed_dict={mainQN.inputs:next_states_batch, mainQN.dropout_ratio:1.0})
        target_Q_values = self.sess.run(targetQN.q_values,feed_dict={targetQN.inputs:next_states_batch, targetQN.dropout_ratio:1.0})

        end_multiplier = -(done_batch - 1) # End multiplier is inverse of done(I.E. if you are done you don't want to find the value of the next state)
        target_Q_values = target_Q_values[range(len(action_batch)),best_actions_index] # Choose the Q-values from the target network to match the action taken by our main network
        targetQ = reward_batch + (self.gamma*target_Q_values*end_multiplier)
                    
        # Train the network
        _ = self.sess.run(mainQN.updateModel,
            feed_dict={mainQN.inputs:states_batch,
                       mainQN.targetQ:targetQ,
                       mainQN.actions:action_batch,
                       mainQN.dropout_ratio:1.0})
        
        updateTarget(self.targetOps, self.sess) #Set the target network to be equal to the primary network.

    def saveStats(self, summary_writer, stats_freq):
            #Periodically save the model. 
            if self.episode_number % stats_freq == 0:

                mean_reward = np.mean(self.reward_per_episode[-stats_freq:])
                mean_length = np.mean(self.steps_per_episode[-stats_freq:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))

                summary_writer.add_summary(summary, self.episode_number)
                print(self.episode_number, "Mean reward: ", mean_reward, " Mean steps: ", int(mean_length), " e: ", self.dropout_ratio)

                summary_writer.flush()

    def saveModel(self, saver, model_freq):
            if self.episode_number % model_freq == 0:
                saver.save(self.sess,path+'/model-'+str(self.episode_number)+'.cptk')
                print("Saved Model")