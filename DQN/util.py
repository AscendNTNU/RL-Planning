import random
import numpy as np
from collections import namedtuple

path = "./dqn"             #The path to save our model to.

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ExperienceBuffer():
    def __init__(self, buffer_size=1000000, batch_size=50):
        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
            
    @property
    def sample(self):
        samples = random.sample(self.buffer, self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
        return states_batch, action_batch, reward_batch, next_states_batch, done_batch
    
    @property
    def full(self):
        return len(self.buffer) == self.buffer_size

    @property
    def fraction_filled(self):
        return len(self.buffer) / self.buffer_size

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

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