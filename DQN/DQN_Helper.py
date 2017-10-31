import random
import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ExperienceBuffer():
    def __init__(self, buffer_size=1000000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop()
        self.buffer.extend(experience)
            
    def sample(self, size):
        samples = random.sample(self.buffer, size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
        return states_batch, action_batch, reward_batch, next_states_batch, done_batch
       
def discountRewards(reward, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_reward = np.zeros_like(reward)
    running_add = 0
    for t in reversed(xrange(0, reward.size)):
        running_add = running_add * gamma + reward[t]
        discounted_reward[t] = running_add
    return discounted_reward

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:int(total_vars/2)]): 
        op_holder.append(tfVars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tfVars[idx+int(total_vars/2)].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def observation_to_input_array(ai_view):
    result = np.array([ai_view.elapsed_time%20, ai_view.drone_x/20.0, ai_view.drone_y/20.0])
    result = np.append(result, ai_view.target_x[0]/20.0)
    result = np.append(result, ai_view.target_y[0]/20.0)
    result = np.append(result, ai_view.target_q[0]/(3.14159265359*2))
    result = np.append(result, ai_view.target_removed[0])
    return result

def target_observation_to_input_array(ai_view, target):
    result = np.array([ai_view.elapsed_time%20, ai_view.drone_x/20.0, ai_view.drone_y/20.0])
    result = np.append(result, ai_view.target_x[target]/20.0)
    result = np.append(result, ai_view.target_y[target]/20.0)
    result = np.append(result, ai_view.target_q[target]/(3.14159265359*2))
    result = np.append(result, ai_view.target_removed[target])
    return result

def chooseRobot(ai_view):
    max_y = 0.0
    target = 0
    for i in range(Num_Targets):
        if(ai_view.target_y[i]>max_y and not ai_view.target_y[i]>20):
            max_y = ai_view.target_y[i]
            target = i
    return target