import os
import tensorflow as tf

from util import *
from sim_variable_setup import *
import network
import controller

if not os.path.exists(path):
    os.makedirs(path)

h_size = 128            #Hidden layer size
learning_rate = 1e-4        

tf.reset_default_graph()

mainQN = network.Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size)
targetQN = network.Qnetwork(lr=learning_rate, s_size=D, a_size=len(action_pool), h_size=h_size) #Load the agent.

summary_writer = tf.summary.FileWriter("dqn_summary")

saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, 0.001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    updateTarget(targetOps, sess)

    controller = controller.Controller(batch_size=1000,
                                buffer_size=1000000,
                                anneling_episodes=200000,
                                update_freq=4,
                                gamma=0.95,
                                sess=sess,
                                targetOps=targetOps)
                
    print("Filling the experience buffer - This can take some time")
    while not controller.experience_buffer.full:
        controller.runEpisode(mainQN)
        print("Progress {:2.1%}".format(controller.experience_buffer.fraction_filled, end="\r"))

    while 1:
        controller.runEpisode(mainQN, targetQN)
        controller.decreaseDropout()
        controller.saveStats(summary_writer, stats_freq=1000)
        controller.saveModel(saver, model_freq=10000)