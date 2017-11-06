from ctypes import *


sim = CDLL('../cppWrapper/PythonAccessToSim.so')

Num_Targets =sim.get_sim_Num_Targets()
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

sim.step.restype = step_result
sim.send_command.restype = c_int
sim.initialize.restype = c_int
sim.recieve_state_gui.restype = step_result
sim.send_command_gui.restype = c_int
sim.action_rewards.restype = c_int