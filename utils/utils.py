# AUTHOR: Farzan Memarian
import numpy as np
import torch
import copy
from pdb import set_trace



# def meta_input(D_in, agent_state, env_state, goal):
# 	'''
# 	returns the meta-controller's input from agent state, env state and goal
# 	'''
# 	meta_input = torch.zeros([2, D_in, D_in], dtype=torch.float32, device=device)
# 	meta_input[0] = goal
# 	meta_input[1] = copy.deepcopy(env_state)
# 	meta_input[1][agent_state[0,0], agent_state[0,1]] = -1
# 	return meta_input

# def meta_input2(D_in, agent_env_state, goal):
# 	'''
# 	returns the meta-controller's input from agent-env state and goal
# 	'''
# 	meta_input = torch.zeros([2, D_in, D_in], dtype=torch.float32, device=device)
# 	meta_input[0] = goal
# 	meta_input[1] = copy.deepcopy(agent_env_state)
# 	return meta_input

# def cntr_input(D_in, agent_state, env_state, goal):
# 	cntr_input = torch.zeros([2, D_in, D_in], dtype=torch.float32, device=device)
# 	cntr_input[0] = goal
# 	cntr_input[1] = copy.deepcopy(env_state)
# 	cntr_input[1][agent_state[0,0], agent_state[0,1]] = -1
# 	return cntr_input	

def agent_env_state(agent_loc, env_state):
	env_state = copy.deepcopy(env_state)
	n,m = env_state.size()
	env_state = torch.unsqueeze(env_state,0)
	agent_state = torch.zeros((1,n,m), device=env_state.device)
	agent_state[0, agent_loc[0,0], agent_loc[0,1]] = 1
	agent_env_state = torch.cat([agent_state, env_state],0)
	return agent_env_state
    
# def agent_env_state_deep(agent_state, env_state):
# 	agent_env_state = copy.deepcopy(env_state)
# 	agent_env_state[agent_state[0,0], agent_state[0,1]] = -1
# 	return torch.unsqueeze(torch.unsqueeze(agent_env_state,0),0).to(device)

def agent_env_state_deep(agent_state, env_state):
	agent_env_state = copy.deepcopy(env_state)
	agent_env_state[agent_state[0,0], agent_state[0,1]] = -1
	return torch.unsqueeze(agent_env_state,0)
