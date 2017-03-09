import numpy as np
from ctypes import *

temp = CDLL('./PythonAccessToSim.so')
Num_Targets = temp.get_sim_Num_Targets()

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

def observation_to_input_array(ai_view):
	#print(ai_view.elapsed_time)
	result = np.array([ai_view.elapsed_time%20/20, ai_view.drone_x/20, ai_view.drone_y/20])
	for i in range(Num_Targets):
		result = np.append(result, ai_view.target_x[i]/20)
		result = np.append(result, ai_view.target_y[i]/20)
		result = np.append(result, ai_view.target_q[i]/(3.14159265359*2))
	return result