# AUTHOR: Farzan Memarian
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import namedtuple
import sys
sys.path.append('../')
from envs.gridworld3 import Gridworld
from agent.agent3 import hDQN, net_CNN, net_MLP
import transformer
from utils import utils
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from pdb import set_trace 
import pickle
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(args):

    NOTE = "reset, transformer, larger gridworld" 

    # *****************
    init_vars = {}
    init_vars["fileName"] = "3_cntr_dev"
    init_vars["pretrained_cntr_folder"] = "3_cntr_v4" # folder name for trained cntr network
    init_vars["pretrained_meta_folder"] = "3_meta_v4" # folder name for trained meta network
    init_vars["main_function"]  = ["train_cntr","train_meta","train_dhrl","test_cntr","test_both"][0]
    init_vars["num_epis"] = 300000
    #  WRITE PERIODS
    init_vars["stat_period_dhrl"] = 50
    init_vars["stat_period_cntr"] = 50
    init_vars["reset_type"] = ["reset","reset_finite","reset_total"][0]
    init_vars["run_mode"] = ["restart","continue"][0]
    init_vars["no_anneal"] = [False, True][0]
    init_vars["verbose"] = [False, True][0]
    init_vars["device"] = ["cpu","cuda:1"][0]
    # *****************
    # GRID WORLD GEOMETRICAL PARAMETERS
    n_dim = 9 # pick odd numbers
    n_obj = 5
    min_num = 1
    max_num = 10
    num_gridworlds = 10 # used only if reset_finite is picked

    # Network type
    cntr_network = ["MLP", "CNN", "transformer"][2] 
    meta_network = ["MLP", "CNN", "transformer"][1]

    # LOSS params
    meta_loss = ["SmoothL1Loss", "MSEloss"][0]
    meta_optimizer = ["Adam", "RMSprop", "SGD"][0]
    meta_lr = 0.0001
    # ------
    cntr_loss = ["SmoothL1Loss", "MSEloss"][0]
    cntr_optimizer = ["Adam", "RMSprop", "SGD"][0]
    cntr_lr = 0.0001

    # extr REWARDS (to be used by the meta controller)
    game_over_reward = -1
    step_reward = 0
    current_goal_reward = 1
    final_goal_reward = 10 #currently not being used

    # int REWARDS (to be used by the controller)
    int_goal_reward = 1
    int_step_reward = 0
    int_wrong_goal_reward = -1

    # CLAMPING
    meta_clamp = False
    cntr_clamp = False

    # PARAMETERS OF META 
    meta_batch_size = 32
    init_vars["meta_eps_start_temp"] = 0.001
    init_vars["meta_eps_end_temp"] = 5
    init_vars["meta_eps_decay_temp"] = 100000
    meta_memory_size = 10000

    # PARAMETERS OF CNTR
    batch_size = 64
    gamma = 0.9
    init_vars["cntr_eps_start_temp"] = 0.001
    init_vars["cntr_eps_end_temp"] = 5
    init_vars["cntr_eps_decay_temp"] = 100000
    tau = 0.001
    cntr_memory_size = 10000
    
    cntr_Transition = namedtuple("cntr_Transition", 
        ["agent_env_cntr", "action_idx", "int_reward", "next_agent_env_cntr", "meta_goal", 
        "next_available_actions", "cntr_done"])
    meta_Transition = namedtuple("meta_Transition",   
        ["agent_env_state_0", "meta_goal", "reward", "next_agent_env_state", 
        "next_available_goals", "terminal", "meta_exp_counter"])

    # create and initialize the environment   
    env = Gridworld(n_dim = n_dim, 
                    n_obj=n_obj, 
                    min_num = min_num, 
                    max_num = max_num,
                    num_gridworlds = num_gridworlds,
                    game_over_reward = game_over_reward, 
                    step_reward = step_reward, 
                    current_goal_reward=current_goal_reward, 
                    final_goal_reward = final_goal_reward,
                    int_goal_reward=int_goal_reward, 
                    int_step_reward=int_step_reward, 
                    int_wrong_goal_reward=int_wrong_goal_reward,
                    reset_type=init_vars["reset_type"],
                    device = torch.device(init_vars["device"])
                    )

    # create and initialize the agent
    agent = hDQN(env=env, 
                batch_size=batch_size,
                meta_batch_size=meta_batch_size, 
                gamma=gamma,
                meta_policy_temp=init_vars["meta_eps_start_temp"], 
                cntr_policy_temp=init_vars["cntr_eps_start_temp"], 
                tau = tau,
                cntr_Transition = cntr_Transition,
                cntr_memory_size = cntr_memory_size,
                meta_Transition = meta_Transition,
                meta_memory_size = meta_memory_size,
                meta_loss = meta_loss,
                meta_optimizer = meta_optimizer,
                meta_lr = meta_lr,
                cntr_loss = cntr_loss,
                cntr_optimizer = cntr_optimizer,
                cntr_lr = cntr_lr,
                meta_clamp = meta_clamp,
                cntr_clamp = cntr_clamp,
                cntr_network_name = cntr_network,
                meta_network_name = meta_network,
                device = torch.device(init_vars["device"])
                )

    params_cntr_run = {
            'NOTE' : NOTE,
            "fileName" : init_vars["fileName"],
            "main_function" : init_vars["main_function"],
            "num_epis" : init_vars["num_epis"],
            "stat_period_dhrl" : init_vars["stat_period_dhrl"],
            "stat_period_cntr" : init_vars["stat_period_cntr"],
            "reset_type" : init_vars["reset_type"], 
            "no_anneal" : init_vars["no_anneal"], 
            "cntr_network" : cntr_network,
            "meta_network" : meta_network,
            "n_dim" : n_dim, 
            "n_obj":n_obj, 
            "min_num" : min_num,
            "max_num" : max_num,
            "num_gridworlds" : num_gridworlds,
            "int_goal_reward":int_goal_reward, 
            "int_step_reward":int_step_reward, 
            "int_wrong_goal_reward":int_wrong_goal_reward,
            "batch_size":batch_size,
            "gamma":gamma,
            "cntr_clamp" : cntr_clamp,
            "cntr_eps_start_temp" : init_vars["cntr_eps_start_temp"],
            "cntr_eps_end_temp" : init_vars["cntr_eps_end_temp"],
            "cntr_eps_decay_temp" : init_vars["cntr_eps_decay_temp"],
            "tau" : tau,
            "cntr_loss" : cntr_loss,
            "cntr_optimizer" : cntr_optimizer,
            "cntr_lr" : cntr_lr,
            "device" : init_vars["device"]
            }

    params_dhrl_run = {
            'NOTE' : NOTE,
            "fileName" : init_vars["fileName"],
            "pretrained_cntr_folder" : init_vars["pretrained_cntr_folder"],
            "main_function" : init_vars["main_function"],
            "num_epis" : init_vars["num_epis"],
            "stat_period_dhrl" : init_vars["stat_period_dhrl"],
            "stat_period_cntr" : init_vars["stat_period_cntr"],
            "reset_type" : init_vars["reset_type"], 
            "no_anneal" : init_vars["no_anneal"], 
            "cntr_network" : cntr_network,
            "meta_network" : meta_network,
            "n_dim" : n_dim, 
            "n_obj":n_obj, 
            "min_num" : min_num,
            "max_num" : max_num,
            "num_gridworlds" : num_gridworlds,
            "game_over_reward" : game_over_reward, 
            "step_reward" : step_reward, 
            "current_goal_reward":current_goal_reward, 
            "final_goal_reward" : final_goal_reward,
            "int_goal_reward":int_goal_reward, 
            "int_step_reward":int_step_reward, 
            "int_wrong_goal_reward":int_wrong_goal_reward,
            "batch_size":batch_size,
            "meta_batch_size":meta_batch_size, 
            "gamma":gamma,
            "meta_clamp" : meta_clamp,
            "cntr_clamp" : cntr_clamp,
            "meta_eps_start_temp" : init_vars["meta_eps_start_temp"],
            "meta_eps_end_temp" : init_vars["meta_eps_end_temp"],
            "meta_eps_decay_temp" : init_vars["meta_eps_decay_temp"],
            "cntr_eps_start_temp" : init_vars["cntr_eps_start_temp"],
            "cntr_eps_end_temp" : init_vars["cntr_eps_end_temp"],
            "cntr_eps_decay_temp" : init_vars["cntr_eps_decay_temp"],
            "tau" : tau,
            "cntr_memory_size" : cntr_memory_size,
            "meta_memory_size" : meta_memory_size,
            "meta_loss" : meta_loss,
            "meta_optimizer" : meta_optimizer,
            "meta_lr" : meta_lr,
            "cntr_loss" : cntr_loss,
            "cntr_optimizer" : cntr_optimizer,
            "cntr_lr" : cntr_lr,
            "device" : init_vars["device"]
            }

    params_meta_run = {
            'NOTE' : NOTE,
            "fileName" : init_vars["fileName"],
            "main_function" : init_vars["main_function"],
            "num_epis" : init_vars["num_epis"],
            "stat_period_dhrl" : init_vars["stat_period_dhrl"],
            "reset_type" : init_vars["reset_type"], 
            "no_anneal" : init_vars["no_anneal"], 
            "meta_network" : meta_network,
            "n_dim" : n_dim, 
            "n_obj":n_obj, 
            "min_num" : min_num,
            "max_num" : max_num,
            "num_gridworlds" : num_gridworlds,
            "game_over_reward" : game_over_reward, 
            "step_reward" : step_reward, 
            "current_goal_reward":current_goal_reward, 
            "final_goal_reward" : final_goal_reward,
            "int_goal_reward":int_goal_reward, 
            "int_step_reward":int_step_reward, 
            "int_wrong_goal_reward":int_wrong_goal_reward,
            "batch_size":batch_size,
            "meta_batch_size":meta_batch_size, 
            "gamma":gamma,
            "meta_clamp" : meta_clamp,
            "meta_eps_start_temp" : init_vars["meta_eps_start_temp"],
            "meta_eps_end_temp" : init_vars["meta_eps_end_temp"],
            "meta_eps_decay_temp" : init_vars["meta_eps_decay_temp"],
            "meta_memory_size" : meta_memory_size,
            "meta_loss" : meta_loss,
            "meta_optimizer" : meta_optimizer,
            "meta_lr" : meta_lr,
            "device" : init_vars["device"]
            }

    logs_address, models_address, _, _ = addresses(init_vars)
    param_file = "params_" + init_vars["main_function"]
    if init_vars["main_function"] == "train_cntr":
        param_var = params_cntr_run
    elif init_vars["main_function"] == "train_meta":
        param_var = params_meta_run
    elif init_vars["main_function"] == "train_dhrl":
        param_var = params_dhrl_run

    create_directories(init_vars)
    with open(logs_address+param_file,'w') as f:
        for key, value in param_var.items():
            f.write('{0}: {1}\n'.format(key, value))
    with open(models_address+param_file,'w') as f:
        for key, value in param_var.items():
            f.write('{0}: {1}\n'.format(key, value))    

    return env, agent, init_vars

def create_directories(init_vars):
    # this function deletes the directory if it already exists 
    # and recreates it. 
    cmd = 'rm -rf ../logs/' + init_vars["fileName"]
    os.system(cmd)
    cmd = 'mkdir ../logs/' + init_vars["fileName"]
    os.system(cmd)
    
    cmd = 'rm -rf ../saved_models/' + init_vars["fileName"]
    os.system(cmd)
    cmd = 'mkdir ../saved_models/' + init_vars["fileName"]
    os.system(cmd)

def addresses(init_vars):
    logs_address = "../logs/" + init_vars["fileName"] + "/"
    models_address = "../saved_models/" + init_vars["fileName"] + "/"
    read_cntr_address = "../saved_models/" + init_vars["pretrained_cntr_folder"] + "/"
    read_meta_address = "../saved_models/" + init_vars["pretrained_meta_folder"] + "/"
    return logs_address, models_address, read_cntr_address, read_meta_address

def train_DHRL(env, agent, init_vars):
    # this should only be run after train_cntr has been run
    device = torch.device(init_vars["device"])

    # creating the folder names
    logs_address, models_address, read_cntr_address, read_meta_address  = addresses(init_vars)
    num_epis = init_vars["num_epis"]
    meta_eps_start_temp = init_vars["meta_eps_start_temp"]
    meta_eps_end_temp = init_vars["meta_eps_end_temp"]
    meta_eps_decay_temp = init_vars["meta_eps_decay_temp"]
    cntr_eps_start_temp = init_vars["cntr_eps_start_temp"]
    cntr_eps_end_temp = init_vars["cntr_eps_end_temp"]
    cntr_eps_decay_temp = init_vars["cntr_eps_decay_temp"]

    # Load cntr models
    if agent.cntr_network_name == "MLP":
        agent.policy_cntr_net = net_MLP(ndim=env.n_dim, out_dim=1).to(device)
        agent.target_cntr_net = net_MLP(ndim=env.n_dim, out_dim=1).to(device)
    elif agent.cntr_network_name == "CNN":
        agent.policy_cntr_net = net_CNN(ndim=env.n_dim, out_dim=1).to(device)
        agent.target_cntr_net = net_CNN(ndim=env.n_dim, out_dim=1).to(device)
    agent.policy_cntr_net.load_state_dict(torch.load(read_cntr_address + "policy_cntr_net_train_cntr.pt"))
    agent.cntr_optimizer, agent.cntr_criterion = agent.set_optim(agent.policy_cntr_net.parameters(), 
            agent.cntr_optimizer_name, agent.cntr_loss_name, agent.cntr_lr_name)   
    agent.target_cntr_net.load_state_dict(torch.load(read_cntr_address + "target_cntr_net_train_cntr.pt"))
    agent.target_cntr_net.eval()

    # Load meta models
    if agent.meta_network_name == "MLP":
        agent.policy_meta_net = net_MLP(ndim=env.n_dim, out_dim=4).to(device)
        agent.target_meta_net = net_MLP(ndim=env.n_dim, out_dim=4).to(device)
    elif agent.meta_network_name == "CNN":
        agent.policy_meta_net = net_CNN(ndim=env.n_dim, out_dim=4).to(device)
        agent.target_meta_net = net_CNN(ndim=env.n_dim, out_dim=4).to(device)
    elif agent.meta_network_name == "transformer":
        agent.policy_meta_net = transformer.att_class(n_dim=env.n_dim, out_dim=4, device=device)
        agent.target_meta_net = transformer.att_class(n_dim=env.n_dim, out_dim=4, device=device)

    agent.policy_meta_net.load_state_dict(torch.load(read_meta_address + "policy_meta_net_train_meta.pt"))
    agent.meta_optimizer, agent.meta_criterion = agent.set_optim(agent.policy_meta_net.parameters(), 
            agent.meta_optimizer_name, agent.meta_loss_name, agent.meta_lr_name)   
    agent.target_meta_net.load_state_dict(torch.load(read_meta_address + "target_meta_net_train_meta.pt"))
    agent.target_meta_net.eval()

    if init_vars["reset_type"] == 'reset': 
        env_state = torch.load(read_cntr_address + "env_state.pt")
        env.env_state = copy.deepcopy(env_state)
        env.env_state_original = copy.deepcopy(env_state)
    elif init_vars["reset_type"] == 'reset_finite':
        pass
        # complete this part

    visits = np.zeros((env.n_dim, env.n_dim))
    game_won_counter = 0
    game_over_counter = 0
    game_result_history = []
    cntr_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    meta_success = 0
    meta_failure = 0
    cntr_success_stats = []
    meta_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached, non_meta_goal_reached, episode_steps]
    hdqn_logs_list = [] # each slement should be [episode_steps, game_won]
    loss_cntr_stat = []
    loss_meta_stat = []
    batch_episode_counter = 0
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")
        agent_loc, env_state = env.reset(init_vars["reset_type"]) 

        visits[agent_loc[0,0].item(), agent_loc[0,1].item()] += 1
        terminal = False
        loss_cntr_epis = 0
        loss_meta_epis = 0
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.select_goal(agent_loc)  # meta cntr selects a goal
            if meta_goal == env.current_target_goal:
                meta_success += 1
            else:
                meta_failure += 1 
            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            total_extr_reward = 0

            agent_env_state_0 = utils.agent_env_state(agent_loc, env_state) # for meta controller start state
            meta_goal_reached = False
            cntr_steps = 0
            while not terminal and not meta_goal_reached: # this loop is for meta, while state not terminal
                cntr_steps += 1
                episode_steps += 1
                action_idx, action = agent.select_action(agent_loc, env_state, meta_goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")
                next_agent_loc, next_env_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                
                extr_reward = torch.tensor([env.extr_reward(next_agent_loc)], device=device)
                int_reward = torch.tensor([env.int_reward(next_agent_loc, meta_goal)], device=device)
                
                i, j = next_agent_loc[0,0].item(), next_agent_loc[0,1].item()
                visits[i, j] += 1
                current_element = int(env.env_state[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                
                if current_element != 0 and not meta_goal_reached:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False
                # print ("state before action ----> " + "[" + str(agent_loc[0,0].item()) +", " + 
                #         str(agent_loc[0,1].item()) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_loc)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached or meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    # object will only be removed if it's the right object to be picked up
                    # else the game ends and it doesn't matter what the remaining objects are
                    # print ("object {} removed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(removed_object))     
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************") 

                if non_meta_goal_reached:
                    cntr_result_history.append([0, episode])
                    cntr_failure += 1
                
                if meta_goal_reached:
                    cntr_result_history.append([1, episode])
                    cntr_success += 1
                    # print("SELECTED GOAL REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the meta goal : " + str(meta_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************")
                
                if game_over:
                    game_over_counter += 1
                    game_result_history.append([0, episode])
                    # print("GAME OVER!!!") 
                    # print("selected goal:" + str(meta_goal))
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))                        
                    # print("********************")
                
                if game_won:
                    game_won_counter += 1
                    game_result_history.append([1, episode])
                    # print("GAME WON!!!") 
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects)) 
                    # print ("current objects: {}".format(env.current_objects))                       
                    # print("********************")


                # if env.env_state[next_agent_loc[0], next_agent_loc[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")
                agent_env_state = utils.agent_env_state(agent_loc, env_state)
                next_agent_env_state = utils.agent_env_state(next_agent_loc, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                exp_cntr = copy.deepcopy([agent_env_state, action_idx, int_reward,  
                    next_agent_env_state, 
                    torch.tensor([meta_goal], dtype=torch.float, device=device), 
                    next_available_actions, cntr_done])
                agent.store(*exp_cntr, meta=False)
                loss_cntr = agent.update(meta=False)
                loss_meta = agent.update(meta=True)
                loss_cntr_epis += loss_cntr
                loss_meta_epis += loss_meta
                total_extr_reward += extr_reward
                agent_loc = copy.deepcopy(next_agent_loc)
                env_state = copy.deepcopy(next_env_state)
                if current_element != 0:
                    cntr_logs_list.append([meta_goal_reached, current_target_reached, cntr_steps, 
                        episode_steps, episode])

            next_agent_env_state = utils.agent_env_state(agent_loc, env_state)
            next_available_goals = env.current_objects
            meta_exp_counter += 1 
            exp_meta = copy.deepcopy([agent_env_state_0, torch.tensor([meta_goal], dtype=torch.float, device=device), total_extr_reward, 
                next_agent_env_state, next_available_goals, terminal, meta_exp_counter])
            agent.store(*exp_meta, meta=True)

        # End of one episode
        hdqn_logs_list.append([episode_steps, game_won, episode])
        print("meta_policy_temp: " + str(agent.meta_policy_temp))
        print("cntr_policy_temp: " + str(agent.cntr_policy_temp))
        loss_cntr_stat.append(loss_cntr_epis/episode_steps)
        loss_meta_stat.append(loss_meta_epis/episode_steps)
        if not init_vars["no_anneal"]:
            # Annealing 
            agent.meta_policy_temp = meta_eps_end_temp + (meta_eps_start_temp - meta_eps_end_temp) * \
                    math.exp(-1. * episode / meta_eps_decay_temp)
            agent.cntr_policy_temp = cntr_eps_end_temp + (cntr_eps_start_temp - cntr_eps_end_temp) * \
                    math.exp(-1. * episode / cntr_eps_decay_temp) 
            # avg_success_rate = cntr_success / goal_selected

            # if(avg_success_rate == 0 or avg_success_rate == 1):
            #     agent.cntr_policy_temp -= anneal_factor
            # elif episode > 200:
            #     agent.cntr_policy_temp = 1 - avg_success_rate
  
            # if(agent.cntr_policy_temp < 0.05):
            #     agent.cntr_policy_temp = 0.05
            # if(agent.meta_policy_temp) < 0.05:
            #     agent.meta_policy_temp = 0.05

        if episode != 0 and episode % (init_vars["stat_period_dhrl"]-1) == 0:
            
            cntr_success_ratio = cntr_success/(cntr_success+cntr_failure)
            cntr_success_stats.append([cntr_success, cntr_failure, goal_selected, cntr_success_ratio, 
                np.mean(loss_cntr_stat), batch_episode_counter])
            loss_cntr_stat = []            
            cntr_success = 0
            cntr_failure = 0
            meta_success_ratio = meta_success/(meta_success+meta_failure)
            meta_success_stats.append([meta_success, meta_failure, goal_selected, meta_success_ratio,
                np.mean(loss_meta_stat), batch_episode_counter])
            loss_meta_stat = []
            meta_success = 0
            meta_failure = 0
            goal_selected = 0
            batch_episode_counter += 1
            print ("SAVING THE LOG FILES .....")
            with open(logs_address + "cntr_success_stats.txt", "w") as file:
                file.write("cntr_success, cntr_failure, goal_selected, success_ratio, cntr_loss, \
                    batch_episode_counter \n")
                for r in cntr_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>14.10f} {5:>3} \n'.format(
                        r[0],r[1],r[2],r[3],r[4],r[5]))

            with open(logs_address + "meta_success_stats.txt", "w") as file:
                file.write("meta_success, meta_failure, goal_selected, success_ratio, meta_loss, \
                    batch_episode_counter \n")
                for r in meta_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>14.10f} {5:>3} \n'.format(
                        r[0],r[1],r[2],r[3],r[4],r[5]))


            with open(logs_address + "game_stats.txt", "w") as file:
                file.write("game_won_counter: {}\n".format(game_won_counter)) 
                file.write("game_over_counter: {}\n".format(game_over_counter))
                file.write("game won ratio: {}\n".format(game_won_counter/(game_over_counter+game_won_counter)))

            with open(logs_address + "game_result_history.txt", "w") as file:
                file.write("game_result, episode \n")
                for result, ep in game_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))
            # *******************************************************
            with open(logs_address + "hdqn_logs_list.txt", "w") as file:
                file.write("episode_steps, game_won, episode \n")
                for line in hdqn_logs_list:
                    file.write('{0:>5} {1:>6} {2:>6} \n'.format(line[0],line[1],line[2]))
            # with open(logs_address + "cntr_logs_list_hdq.txt", "w") as file:
            #     file.write("meta_goal_reached, current_target_reached, cntr_steps, episode_steps, episode \n")
            #     for line in cntr_logs_list:
            #             file.write('{0:>6} {1:>6} {2:>6} {3:>6} {4:>6} \n'.format(line[0],
            #                 line[1],line[2],line[3],line[4]))



            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_meta_net.state_dict(), models_address + "policy_meta_net.pt")
            torch.save(agent.target_meta_net.state_dict(), models_address + "target_meta_net.pt")
            torch.save(agent.policy_cntr_net.state_dict(), models_address + "policy_cntr_net.pt")
            torch.save(agent.target_cntr_net.state_dict(), models_address + "target_cntr_net.pt")


def train_meta(env, agent, init_vars):
    # this function assumes that any object picked by the meta controller is always pickec 
    # bypassing the controller
    device = torch.device(init_vars["device"])

    # creating the folder names
    logs_address, models_address, read_cntr_address, read_meta_address = addresses(init_vars)
    num_epis = init_vars["num_epis"]
    meta_eps_start_temp = init_vars["meta_eps_start_temp"]
    meta_eps_end_temp = init_vars["meta_eps_end_temp"]
    meta_eps_decay_temp = init_vars["meta_eps_decay_temp"]

    visits = np.zeros((env.n_dim, env.n_dim))
    game_won_counter = 0
    game_over_counter = 0
    game_result_history = []
    cntr_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    meta_success = 0
    meta_failure = 0
    cntr_success_stats = []
    meta_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached, non_meta_goal_reached, episode_steps]
    hdqn_logs_list = [] # each slement should be [episode_steps, game_won]
    loss_cntr_stat = []
    loss_meta_stat = []
    batch_episode_counter = 0
    start_stat_period_time = time.time()
    update_stat_period_time_accum = 0
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")

        agent_loc, env_state = env.reset(init_vars["reset_type"]) 

        terminal = False
        loss_meta_epis = 0
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.select_goal(agent_loc)  # meta cntr selects a goal
            if meta_goal == env.current_target_goal:
                meta_success += 1

            else:
                meta_failure += 1 

            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            total_extr_reward = 0

            agent_env_state_0 = utils.agent_env_state(agent_loc, env_state) # for meta controller start state
            meta_goal_reached = False
            while not terminal and not meta_goal_reached: # this loop is for meta, while state not terminal
                episode_steps += 1
                next_agent_loc, next_env_state = env.get_to_object(meta_goal) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                extr_reward = torch.tensor([env.extr_reward(next_agent_loc)], device=device)
                i, j = next_agent_loc[0,0].item(), next_agent_loc[0,1].item()
                current_element = int(env.env_state[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                
                if current_element != 0 and not meta_goal_reached:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False
                # print ("state before action ----> " + "[" + str(agent_loc[0,0].item()) +", " + 
                #         str(agent_loc[0,1].item()) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_loc)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached or meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    # object will only be removed if it's the right object to be picked up
                    # else the game ends and it doesn't matter what the remaining objects are
                    # print ("object {} removed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(removed_object))     
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************") 

 

                
                if game_over:
                    game_over_counter += 1
                    game_result_history.append([0, episode])
                    # print("GAME OVER!!!") 
                    # print("selected goal:" + str(meta_goal))
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))                        
                    # print("********************")
                
                if game_won:
                    game_won_counter += 1
                    game_result_history.append([1, episode])
                    # print("GAME WON!!!") 
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects)) 
                    # print ("current objects: {}".format(env.current_objects))                       
                    # print("********************")


                # if env.env_state[next_agent_loc[0], next_agent_loc[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")
                agent_env_state = utils.agent_env_state(agent_loc, env_state)
                next_agent_env_state = utils.agent_env_state(next_agent_loc, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal

                start_update_time = time.time()
                loss_meta = agent.update(meta=True)
                update_time = time.time() - start_update_time
                update_stat_period_time_accum += update_time

                loss_meta_epis += loss_meta
                total_extr_reward += extr_reward
                agent_loc = copy.deepcopy(next_agent_loc)
                env_state = copy.deepcopy(next_env_state)

            next_agent_env_state = utils.agent_env_state(agent_loc, env_state)
            next_available_goals = env.current_objects
            meta_exp_counter += 1 
            exp_meta = copy.deepcopy([agent_env_state_0, torch.tensor([meta_goal], dtype=torch.float, device=device), total_extr_reward, 
                next_agent_env_state, next_available_goals, terminal, meta_exp_counter])
            agent.store(*exp_meta, meta=True)

        # End of one episode
        hdqn_logs_list.append([episode_steps, game_won, episode])
        print("meta_policy_temp: " + str(agent.meta_policy_temp))
        print("cntr_policy_temp: " + str(agent.cntr_policy_temp))
        loss_meta_stat.append(loss_meta_epis/episode_steps)
        if not init_vars["no_anneal"]:
            # Annealing 
            agent.meta_policy_temp = meta_eps_end_temp + (meta_eps_start_temp - meta_eps_end_temp) * \
                    math.exp(-1. * episode / meta_eps_decay_temp)

            # avg_success_rate = cntr_success / goal_selected

            # if(avg_success_rate == 0 or avg_success_rate == 1):
            #     agent.cntr_policy_temp -= anneal_factor
            # elif episode > 200:
            #     agent.cntr_policy_temp = 1 - avg_success_rate
  
            # if(agent.cntr_policy_temp < 0.05):
            #     agent.cntr_policy_temp = 0.05
            # if(agent.meta_policy_temp) < 0.05:
            #     agent.meta_policy_temp = 0.05

        if episode != 0 and episode % (init_vars["stat_period_dhrl"]-1) == 0:
            
            elapsed_stat_period_time = time.time() - start_stat_period_time
            meta_success_ratio = meta_success/(meta_success+meta_failure)
            meta_success_stats.append([meta_success, meta_failure, goal_selected, meta_success_ratio,
                np.mean(loss_meta_stat), batch_episode_counter, update_stat_period_time_accum,
                elapsed_stat_period_time])
            loss_meta_stat = []
            meta_success = 0
            meta_failure = 0
            goal_selected = 0
            batch_episode_counter += 1
            print ("SAVING THE LOG FILES .....")
            with open(logs_address + "meta_success_stats.txt", "w") as file:
                file.write("meta_success, meta_failure, goal_selected, success_ratio, meta_loss, \
                    batch_episode_counter,update_stat_period_time_accum, elapsed_stat_period_time \n")
                for r in meta_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>14.10f} {5:>3} {6:>6.2f} {7:>6.2f} \n'.format(
                        r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]))

            with open(logs_address + "game_stats.txt", "w") as file:
                file.write("game_won_counter: {}\n".format(game_won_counter)) 
                file.write("game_over_counter: {}\n".format(game_over_counter))
                file.write("game won ratio: {}\n".format(game_won_counter/(game_over_counter+game_won_counter)))

            with open(logs_address + "game_result_history.txt", "w") as file:
                file.write("game_result, episode \n")
                for result, ep in game_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))

            with open(logs_address + "hdqn_logs_list.txt", "w") as file:
                file.write("episode_steps, game_won, episode \n")
                for line in hdqn_logs_list:
                    file.write('{0:>5} {1:>6} {2:>6} \n'.format(line[0],line[1],line[2]))

            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_meta_net.state_dict(), models_address + "policy_meta_net_train_meta.pt")
            torch.save(agent.target_meta_net.state_dict(), models_address + "target_meta_net_train_meta.pt")

            start_stat_period_time = time.time()
            update_stat_period_time_accum = 0

def train_cntr(env, agent, init_vars):

    device = torch.device(init_vars["device"])
    # creating the folders
    logs_address, models_address, _, _ = addresses(init_vars)
    num_epis = init_vars["num_epis"]
    meta_eps_start_temp = init_vars["meta_eps_start_temp"]
    meta_eps_end_temp = init_vars["meta_eps_end_temp"]
    meta_eps_decay_temp = init_vars["meta_eps_decay_temp"]
    cntr_eps_start_temp = init_vars["cntr_eps_start_temp"]
    cntr_eps_end_temp = init_vars["cntr_eps_end_temp"]
    cntr_eps_decay_temp = init_vars["cntr_eps_decay_temp"]

    # save the gridworld to be used for eval later
    if init_vars["reset_type"] == "reset":
        torch.save(env.env_state_original, models_address + "env_state.pt")
    elif init_vars["reset_type"] == "reset_finite":
        torch.save(env.env_state_finite_store, models_address + "env_state_finite_store.pt")

    visits = np.zeros((env.n_dim, env.n_dim))
    game_won_counter = 0
    game_over_counter = 0
    cntr_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    cntr_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached]
    loss_cntr_stat = []
    batch_episode_counter = 0
    start_stat_period_time = time.time()
    update_stat_period_time_accum = 0
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")
        agent_loc, env_state = env.reset(init_vars["reset_type"]) 
        visits[agent_loc[0,0].item(), agent_loc[0,1].item()] += 1
        terminal = False
        non_meta_goal_reached = False
        loss_cntr_epis = 0
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.random_goal_selection()  # meta cntr selects a goal
            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            meta_goal_reached = False
            cntr_steps = 0
            while not terminal and not meta_goal_reached: # this loop is for cntr, while state not terminal
                cntr_steps += 1
                episode_steps += 1
                action_idx, action, action_probs = agent.select_action(agent_loc, env_state, meta_goal) # cntr selects an action among permitable actions
                # print ("---------- action_probs: {}".format(action_probs))
                # print(str((state,action)) + "; ")

                next_agent_loc, next_env_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 

                int_reward = env.int_reward(next_agent_loc, meta_goal)
                
                i, j = next_agent_loc[0,0].item(), next_agent_loc[0,1].item()
                visits[i, j] += 1
                current_element = int(env.env_state[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                if current_element != 0 and not meta_goal_reached:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False
                if init_vars["verbose"]:
                    print ("state before action ----> " + "[" + str(agent_loc[0,0].item()) +", " + 
                            str(agent_loc[0,1].item()) + "]" )
                    print ("action ----> "  + action)
                    print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                    print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_loc)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached or meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    # object will only be removed if it's the right object to be picked up
                    # else the game ends and it doesn't matter what the remaining objects are
                    # print ("object {} removed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(removed_object))     
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************") 

                if non_meta_goal_reached:
                    cntr_result_history.append([0, episode])
                    cntr_failure += 1
                
                if meta_goal_reached:
                    cntr_result_history.append([1, episode])
                    cntr_success += 1
                    # print("SELECTED GOAL REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the meta goal : " + str(meta_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************")

                if game_over:
                    game_over_counter += 1
                    # print("GAME OVER!!!") 
                    # print("selected goal:" + str(meta_goal))
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))                        
                    # print("********************")
                
                if game_won:
                    game_won_counter += 1
                    # print("GAME WON!!!") 
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects)) 
                    # print ("current objects: {}".format(env.current_objects))                       
                    # print("********************")


                # if env.env_state[next_agent_loc[0], next_agent_loc[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")

                agent_env_state = utils.agent_env_state(agent_loc, env_state)
                next_agent_env_state = utils.agent_env_state(next_agent_loc, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                exp_cntr = copy.deepcopy([agent_env_state, action_idx, int_reward,  
                    next_agent_env_state, torch.tensor([meta_goal], dtype=torch.float, device=device), 
                    next_available_actions, cntr_done])
                agent.store(*exp_cntr, meta=False)
                start_update_time = time.time()
                loss_cntr = agent.update(meta=False)
                update_time = time.time() - start_update_time
                update_stat_period_time_accum += update_time
                loss_cntr_epis += loss_cntr
                agent_loc = copy.deepcopy(next_agent_loc)
                env_state = copy.deepcopy(next_env_state)

                if current_element != 0:
                    cntr_logs_list.append([meta_goal_reached, current_target_reached, cntr_steps,  
                        episode_steps, episode])



        #  THIS IS THE END OF ONE EPISODE
        print("cntr_policy_temp: " + str(agent.cntr_policy_temp))
        loss_cntr_stat.append(loss_cntr_epis/episode_steps)
        if not init_vars["no_anneal"]:
            # Annealing 
            agent.cntr_policy_temp = cntr_eps_end_temp + (cntr_eps_start_temp - cntr_eps_end_temp) * \
                    math.exp(-1. * episode / cntr_eps_decay_temp)
            # if episode < num_epis / 10:
            #     agent.cntr_policy_temp -= anneal_factor * init_vars["stat_period_cntr"]
            # else:
            #     success_ratio = cntr_success / (cntr_success + cntr_failure)
            #     if success_ratio == 1 or success_ratio == 0:
            #         agent.cntr_policy_temp -= anneal_factor
            #     else:
            #         agent.cntr_policy_temp = 1 - success_ratio
            
            # if agent.cntr_policy_temp < 0.05:
            #     agent.cntr_policy_temp = 0.05



        if episode != 0 and episode % (init_vars["stat_period_cntr"]-1) == 0:
            elapsed_stat_period_time = time.time() - start_stat_period_time

            success_ratio = cntr_success/(cntr_success+cntr_failure)
            cntr_success_stats.append([cntr_success,cntr_failure,goal_selected, success_ratio, 
                np.mean(loss_cntr_stat), batch_episode_counter, update_stat_period_time_accum, 
                elapsed_stat_period_time])
            loss_cntr_stat = []
            batch_episode_counter += 1
            cntr_success = 0
            cntr_failure = 0
            goal_selected = 0
            print ("SAVING THE LOG FILES .....")
            with open(logs_address + "cntr_success_stats.txt", "w") as file:
                file.write("cntr_success, cntr_failure, goal_selected, success_ratio, cntr_loss, \
                    batch_episode_counter, update_stat_period_time_accum(secs), elapsed_stat_period_time\n")
                for r in cntr_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>14.10f} {5:>3} {6:>6.2f} {7:>6.2f}\n'.format(
                        r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]))


            with open(logs_address + "cntr_result_history.txt", "w") as file:
                file.write("cntr_result, episode \n")
                for result, ep in cntr_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))


            with open(logs_address + "cntr_logs_list.txt", "w") as file:
                file.write("meta_goal_reached, current_target_reached, cntr_steps, episode_steps, episode \n")
                for line in cntr_logs_list:
                        file.write('{0:>6} {1:>6} {2:>6} {3:>6} {4:>6} \n'.format(line[0],
                            line[1],line[2],line[3],line[4]))

            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_cntr_net.state_dict(), models_address + "policy_cntr_net_train_cntr.pt")
            torch.save(agent.target_cntr_net.state_dict(), models_address + "target_cntr_net_train_cntr.pt")

            start_stat_period_time = time.time()
            update_stat_period_time_accum = 0



def test_cntr(env, agent, init_vars):

    device = torch.device(init_vars["device"])
    logs_address, models_address, read_cntr_address, _ = addresses(init_vars)
    num_epis = init_vars["num_epis"]
    meta_eps_start_temp = init_vars["meta_eps_start_temp"]
    meta_eps_end_temp = init_vars["meta_eps_end_temp"]
    meta_eps_decay_temp = init_vars["meta_eps_decay_temp"]
    cntr_eps_start_temp = init_vars["cntr_eps_start_temp"]
    cntr_eps_end_temp = init_vars["cntr_eps_end_temp"]
    cntr_eps_decay_temp = init_vars["cntr_eps_decay_temp"]

    # Load models
    if agent.cntr_network_name == "CNN":
        agent.policy_cntr_net = cntr_net_CNN(ndim=env.n_dim).to(device)
        agent.target_cntr_net = cntr_net_CNN(ndim=env.n_dim).to(device)
    if agent.cntr_network_name == "MLP":
        agent.policy_cntr_net = cntr_net_MLP(ndim=env.n_dim).to(device)
        agent.target_cntr_net = cntr_net_MLP(ndim=env.n_dim).to(device)
    
    agent.policy_cntr_net.load_state_dict(torch.load(read_cntr_address + "policy_cntr_net_train_cntr.pt"))
    agent.cntr_optimizer, agent.cntr_criterion = agent.set_optim(agent.policy_cntr_net.parameters(), 
            agent.cntr_optimizer_name, agent.cntr_loss_name, agent.cntr_lr_name)   
    agent.target_cntr_net.load_state_dict(torch.load(read_cntr_address + "target_cntr_net_train_cntr.pt"))
    agent.target_cntr_net.eval()

    if init_vars["reset_type"] == 'reset': 
        env_state = torch.load(read_cntr_address + "env_state.pt")
        env.env_state = copy.deepcopy(env_state)
        env.env_state_original = copy.deepcopy(env_state)

    visits = np.zeros((env.n_dim, env.n_dim))
    game_won_counter = 0
    game_over_counter = 0
    cntr_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    cntr_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached]
    loss_cntr_stat = []
    batch_episode_counter = 0

    agent.cntr_policy_temp = 0.05 # fixed at a small number
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")

        agent_loc, env_state = env.reset(init_vars["reset_type"]) 

        visits[agent_loc[0,0].item(), agent_loc[0,1].item()] += 1
        terminal = False
        non_meta_goal_reached = False
        loss_cntr_epis = 0
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.random_goal_selection()  # meta cntr selects a goal
            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            meta_goal_reached = False
            cntr_steps = 0
            while not terminal and not meta_goal_reached: # this loop is for cntr, while state not terminal
                cntr_steps += 1
                episode_steps += 1
                action_idx, action = agent.select_action(agent_loc, env_state, meta_goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")

                next_agent_loc, next_env_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 

                int_reward = env.int_reward(next_agent_loc, meta_goal)
                
                i, j = next_agent_loc[0,0].item(), next_agent_loc[0,1].item()
                visits[i, j] += 1
                current_element = int(env.env_state[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                if current_element != 0 and not meta_goal_reached:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False
                # print ("state before action ----> " + "[" + str(agent_loc[0,0].item()) +", " + 
                #         str(agent_loc[0,1].item()) + "]" )
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_loc)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached or meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    
                    # object will only be removed if it's the right object to be picked up
                    # else the game ends and it doesn't matter what the remaining objects are
                    # print ("object {} removed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(removed_object))     
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************") 

                if non_meta_goal_reached:
                    cntr_result_history.append([0, episode])
                    cntr_failure += 1
                
                if meta_goal_reached:
                    cntr_result_history.append([1, episode])
                    cntr_success += 1
                    # print("SELECTED GOAL REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the meta goal : " + str(meta_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))
                    # print ("********************")

                if game_over:
                    game_over_counter += 1
                    # print("GAME OVER!!!") 
                    # print("selected goal:" + str(meta_goal))
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    # print ("current objects: {}".format(env.current_objects))                        
                    # print("********************")
                
                if game_won:
                    game_won_counter += 1
                    # print("GAME WON!!!") 
                    # print("the object reached: " +  str(current_element))
                    # print("the current target goal: " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects)) 
                    # print ("current objects: {}".format(env.current_objects))                       
                    # print("********************")


                # if env.env_state[next_agent_loc[0], next_agent_loc[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")

                agent_env_state = utils.agent_env_state(agent_loc, env_state)
                next_agent_env_state = utils.agent_env_state(next_agent_loc, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                # exp_cntr = copy.deepcopy([agent_env_state, action_idx, int_reward,  
                #     next_agent_env_state, torch.tensor([meta_goal], dtype=torch.float, device=device), 
                #     next_available_actions, cntr_done])
                # agent.store(*exp_cntr, meta=False)
                # loss_cntr = agent.update(meta=False)

                # loss_cntr_epis += loss_cntr

                agent_loc = copy.deepcopy(next_agent_loc)
                env_state = copy.deepcopy(next_env_state)

                if current_element != 0:
                    cntr_logs_list.append([meta_goal_reached, current_target_reached, cntr_steps,  
                        episode_steps, episode])



        #  THIS IS THE END OF ONE EPISODE
        print("cntr_policy_temp: " + str(agent.cntr_policy_temp))
        # loss_cntr_stat.append(loss_cntr_epis/episode_steps)


        if episode != 0 and episode % (init_vars["stat_period_cntr"]-1) == 0:

            success_ratio = cntr_success/(cntr_success+cntr_failure)
            cntr_success_stats.append([cntr_success,cntr_failure,goal_selected, success_ratio, 
                 batch_episode_counter])
            loss_cntr_stat = []
            batch_episode_counter += 1
            cntr_success = 0
            cntr_failure = 0
            goal_selected = 0
            print ("SAVING THE LOG FILES .....")
            with open(logs_address + "cntr_success_stats.txt", "w") as file:
                file.write("cntr_success, cntr_failure, goal_selected, success_ratio, \
                    batch_episode_counter \n")
                for r in cntr_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>3} \n'.format(
                        r[0],r[1],r[2],r[3],r[4]))

            with open(logs_address + "cntr_result_history.txt", "w") as file:
                file.write("cntr_result, episode \n")
                for result, ep in cntr_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))

            with open(logs_address + "cntr_logs_list.txt", "w") as file:
                file.write("meta_goal_reached, current_target_reached, cntr_steps, episode_steps, episode \n")
                for line in cntr_logs_list:
                        file.write('{0:>6} {1:>6} {2:>6} {3:>6} {4:>6} \n'.format(line[0],
                            line[1],line[2],line[3],line[4]))

def test_both():

    pass

def main(args):

    env, agent, init_vars = init(args)

    if init_vars["main_function"] == "train_cntr" :
        train_cntr(env, agent, init_vars)
    elif init_vars["main_function"] == "train_dhrl" :
        train_DHRL(env, agent, init_vars)
    elif init_vars["main_function"] == "train_meta" :
        train_meta(env, agent, init_vars)
    elif init_vars["main_function"] == "test_cntr":
        test_cntr(env, agent, init_vars)
    elif init_vars["main_function"] == "test_both":
        test_both(env, agent, init_vars)
    
            
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Relational-RL')
    parser.add_argument('--fileName',type=str, default="0")
    parser.add_argument('--train_cntr ', default=False)

    parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                        help='resume from model stored')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    args = parser.parse_args()

    main(args)