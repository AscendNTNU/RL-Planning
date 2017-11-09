#define SIM_IMPLEMENTATION
#define SIM_CLIENT_CODE
#include "../sim.h"
#include "../gui.h"
#include <stdio.h>
#include <iostream>
//This program is used to access the simulation from python. Needs to be compiled as a .dll for windows or .so for linux to work.
//The three functions used in python are initialize(), step() and send_command()
//step returns the observation, reward and 1 if done.

int step_length = 60*5; //Frames?
int episode_length = 200;
int last_robot_reward = 0;
int last_time = 0;
int last_position_reward = 0;
int reward_for_robot = 1000;
double last_robot_q[Num_Targets];
static double GRID[22][22];
const float SPEED = 0.33;
//AI actions
enum ai_Actions{
	ai_Search = 0,
	ai_LandOnTop,
	ai_Track,
	ai_Wait
};

struct Plank
{
    float x_1;
    float y_1;
    float x_2;
    float y_2;
};


//Input data for neural network
struct ai_data_input_struct{
	float elapsed_time;
	float drone_x;
	float drone_y;
	float target_x[Num_Targets];
	float target_y[Num_Targets];
	float target_q[Num_Targets];
	int target_removed[Num_Targets];
};

struct step_result{
	ai_data_input_struct observation;
	float reward;
	bool done;
	bool cmd_done;
};

ai_data_input_struct ai_data_input;
sim_State state = {};
sim_Observed_State observed_state;
sim_Observed_State prev_obv_state;
sim_Command cmd;

extern "C"{

	int get_sim_Num_Targets(){
		// createGrid();
		return Num_Targets;
	}

	int initialize(){
		//std::cout << "init" << std::endl;
		state = sim_init(rand());
		observed_state = sim_observe_state(state);
		//for reward calculation
		last_robot_reward = 0;
		last_time = 0;

		return 0;
	}

	float reward_calculator(){
		float result = 0;
    	float reward = 0;

    	reward += observed_state.target_reward[0];
   		result = reward_for_robot*(reward);

    	result -= last_robot_reward;
    	last_robot_reward = reward_for_robot*(reward);
    	//Time spent rewards
    	// result -= (observed_state.elapsed_time - last_time);
    	// last_time = observed_state.elapsed_time;

    	return result/reward_for_robot;
	}

	bool ready_for_command(){
    	if(observed_state.drone_cmd_done){
    	    return true;
    	}
    	return false;
	}

	int get_done(){
		int result = 0;
    	for(int i = 0; i < Num_Targets; i++){
    	    result += observed_state.target_removed[i];
    	}
		if(result == Num_Targets || observed_state.elapsed_time > episode_length){
		// if(observed_state.target_removed[0] || observed_state.elapsed_time > 200){
    	    return 1;
    	}
    	return 0;
	}

	bool targetIsMoving(int target, sim_Observed_State previous_state, sim_Observed_State observed_state){
	    bool moving = true;
	    if (previous_state.target_x[target] == observed_state.target_x[target] && 
	        previous_state.target_y[target] == observed_state.target_y[target]) 
	    {
	        moving = false;
	    }
    	return moving;
	}

	ai_data_input_struct update_ai_input(){
		ai_data_input_struct result;
		result.elapsed_time = observed_state.elapsed_time;
		result.drone_x = observed_state.drone_x;
		result.drone_y = observed_state.drone_y;


		for(int i = 0; i < Num_Targets; i++){
		    //if(observed_state.target_in_view[i]){
			if(observed_state.target_removed[i]){
	    		result.target_x[i] = observed_state.target_x[i];
	    		result.target_y[i] = observed_state.target_y[i];			
			}
			else{
	    		result.target_x[i] = observed_state.target_x[i];
	    		result.target_y[i] = observed_state.target_y[i];
	    	}
		    if(targetIsMoving(i,prev_obv_state,observed_state)){
		    	result.target_q[i] = observed_state.target_q[i];
		    	last_robot_q[i] = observed_state.target_q[i];
		    }

		    else{
		    	result.target_q[i] = (last_robot_q[i]);
		    }
		    result.target_removed[i] = observed_state.target_removed[i];
		}

		return result;
	}

	step_result step(){
	    step_result result;
	    prev_obv_state = observed_state;
	    int tick = 1;
	    while(tick<step_length || !state.drone.cmd_done){
	    	state = sim_tick(state, cmd);
	    	tick += 1;
	    }
	    observed_state = sim_observe_state(state);
	    result.observation = update_ai_input();
	    result.reward = reward_calculator();

	    result.done = get_done();

	    return result;
	}

	int send_command(int a){

		int action_type = a%3;
		if(observed_state.target_removed[a/3]){
			cmd.type = sim_CommandType_NoCommand;
			return 0;
		}

		switch(action_type){
			case 0:
	            cmd.type = sim_CommandType_LandInFrontOf;
	            cmd.i = a/3;
	        break;

	        case 1:
	            cmd.type = sim_CommandType_LandOnTopOf;
	            cmd.i = a/3;
	        break;

	        case 2:
	        	cmd.type = sim_CommandType_Search;
	  			cmd.x = observed_state.target_x[a/3];
	  			cmd.y = observed_state.target_y[a/3];
			break;

	        default:
	        	std::cout << "This shouldn't happen" << std::endl;
	        break;
		}
		state = sim_tick(state, cmd);
		cmd.type = sim_CommandType_NoCommand;
		return 0;
	}

	int action_rewards(int a){
		int target = a/3;
		int reward = 0;
		if(observed_state.target_removed[target]){
			reward -= 1;
		}

		return reward;
	}

	// GUI VERSION:
	int initialize_gui(){
	    sim_init_msgs(true);
	}

	step_result recieve_state_gui(){

		step_result result;	
	    sim_recv_state(&state);
	    prev_obv_state = observed_state;
	    observed_state = sim_observe_state(state);
	    
	    result.observation = update_ai_input();
	    result.reward = reward_calculator();
	    result.done = get_done();
	    result.cmd_done = ready_for_command();
	    return result;
	}


	int send_command_gui(int a)
	{
		sim_Command cmd;

		int action_type = a%3;
		//std::cout << action_type << std::endl;

		if(observed_state.target_removed[a/3]){
			cmd.type = sim_CommandType_NoCommand;
			return 0;
		}

		switch(action_type){
			case 0:
	            cmd.type = sim_CommandType_LandInFrontOf;
	            cmd.i =  a/3;
	            sim_send_cmd(&cmd);
	            //std::cout<<observed_state.elapsed_time<<std::endl;
	            //printf("InFront\n");
	        break;

	        case 1:
	            cmd.type = sim_CommandType_LandOnTopOf;
	            cmd.i =  a/3;
				sim_send_cmd(&cmd);
	            //std::cout<<observed_state.elapsed_time<<std::endl;
				//printf("Ontop\n");
	        break;

	        case 2:
	        	cmd.type = sim_CommandType_Search;
	  			cmd.x = observed_state.target_x[ a/3];//a/2-1];
	  			cmd.y = observed_state.target_y[ a/3];//a/2-1];
	        	sim_send_cmd(&cmd);
	        break;
	        default:
	        	std::cout<<"this shouldn't happen"<<std::endl;

	    }
	    return 0;
	}

}