import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op
from helper import *

class PolicyMonitor(object):
  """
  Helps evaluating a policy by running an episode in an environment,
  saving a video, and plotting summaries to Tensorboard.
  Args:
    env: environment to run in
    policy_net: A policy estimator
    summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
  """
  def __init__(self, env, policy_net, summary_writer, saver=None):


    self.global_policy_net = policy_net
    self.summary_writer = summary_writer
    self.saver = saver
    #self.sp = StateProcessor()

    self.env = CDLL('./PythonAccessToSim.so')
    self.env.step.restype = step_result
    self.env.send_command.restype = c_int
    self.env.initialize.restype = c_int
    self.env.recieve_state_gui.restype = step_result

    self.actions = list(range(0,3*Num_Targets))

    self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "./checkpoints/model"))
    print(self.checkpoint_path)
    # Local policy net
    with tf.variable_scope("policy_eval"):
      self.policy_net = PolicyEstimator(policy_net.num_outputs)

    # Op to copy params from global policy/value net parameters
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.policy_net.states: [state] }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def eval_once(self, sess):
    with sess.as_default(), sess.graph.as_default():
      # Copy params to local model
      global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

      # Run an episode
      done = False
      #print("start")
      self.env.initialize()
      run = self.env.step()
      state = observation_to_input_array(run.ai_data_input)
      total_reward = 0.0
      episode_length = 0
      while not run.done:
        action_probs = self._policy_net_predict(state, sess)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        self.env.send_command(self.actions[action])
        next_run = self.env.step()
        while not next_run.cmd_done:
          next_run = self.env.step()
          total_reward += next_run.reward
        next_state = observation_to_input_array(next_run.ai_data_input)
        total_reward += next_run.reward
        episode_length += 1
        state = next_state
        run = next_run

      #print("done")

      # Add summaries
      episode_summary = tf.Summary()
      episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
      episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
      self.summary_writer.add_summary(episode_summary, global_step)
      self.summary_writer.flush()

      if self.saver is not None:
        self.saver.save(sess, self.checkpoint_path)
        #print("here")

      tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}".format(global_step, total_reward, episode_length))

      return total_reward, episode_length

  def continuous_eval(self, eval_every, sess, coord):
    """
    Continuously evaluates the policy every [eval_every] seconds.
    """
    try:
      while not coord.should_stop():
        self.eval_once(sess)
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      print("shit")
      return