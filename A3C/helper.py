import numpy as np

def observation_to_input_array(ai_view):
	#print(ai_view.elapsed_time)
	result = np.array([ai_view.elapsed_time%20, ai_view.drone_x/20, ai_view.drone_y/20])
	for i in range(Num_Targets):
		result = np.append(result, ai_view.target_x[i]/20)
		result = np.append(result, ai_view.target_y[i]/20)
		result = np.append(result, ai_view.target_q[i]/(3.14159265359*2))
	return result