import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import namedtuple
import sys
sys.path.append('../')
from envs.gridworld3 import Gridworld
from agent.agent3 import hDQN
import transformer
from utils import utils
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from pdb import set_trace 
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(args):
    

    # *****************
    init_vars = {}
    init_vars["fileName"] = "3_cntr_v4"
    init_vars["train_only_cntr"]  = True
    init_vars["num_epis"] = 300000
    #  WRITE PERIODS
    init_vars["stat_period_dhrl"] = 200 
    init_vars["stat_period_cntr"] = 200
    init_vars["reset_type"] = "reset_total" # pick btw "reset" and "reset_total"
    init_vars["no_anneal"] = False
    # *****************

    NOTE = 'the cntr epsilon is annealed every stat_period_dhrl episodes, no fancy annealing program'


    # GRID WORLD GEOMETRICAL PARAMETERS
    D_in = 5 # pick odd numbers
    n_obj = 3
    min_num = 1
    max_num = 3

    # extr REWARDS
    game_over_reward = -10
    step_reward = 0
    current_goal_reward = 10
    final_goal_reward = 100

    # int REWARDS
    int_goal_reward = 200
    int_step_reward = -1
    int_wrong_goal_reward = -200

    # PARAMETERS OF META 
    meta_batch_size = 35
    meta_epsilon = 1
    meta_memory_size = 10000

    # PARAMETERS OF CNTR
    batch_size = 35
    gamma = 0.975
    cntr_epsilon = 1
    tau = 0.001
    cntr_memory_size = 10000


    # LOSS prams
    # pick between MSEloss, SmoothL1Loss
    # pick btw, SGD, RMSprop, Adam
    meta_loss = 'SmoothL1Loss' 
    meta_optimizer = 'RMSprop'  
    meta_lr = 0.0001
    # ------
    cntr_loss = 'SmoothL1Loss'
    cntr_optimizer = 'RMSprop'
    cntr_lr = 0.0001

    cntr_Transition = namedtuple("cntr_Transition", 
        ["agent_env_cntr", "action_idx", "int_reward", "next_agent_env_cntr", "meta_goal", 
        "next_available_actions", "cntr_done"])
    meta_Transition = namedtuple("meta_Transition",   
        ["agent_env_state_0", "meta_goal", "reward", "next_agent_env_state", 
        "next_available_goals", "terminal", "meta_exp_counter"])

    # create and initialize the environment   
    env = Gridworld(D_in = D_in, 
                    n_obj=n_obj, 
                    min_num = min_num, 
                    max_num = max_num,
                    game_over_reward = game_over_reward, 
                    step_reward = step_reward, 
                    current_goal_reward=current_goal_reward, 
                    final_goal_reward = final_goal_reward,
                    int_goal_reward=int_goal_reward, 
                    int_step_reward=int_step_reward, 
                    int_wrong_goal_reward=int_wrong_goal_reward)

    # create and initialize the agent
    agent = hDQN(env=env, 
                batch_size=batch_size,
                meta_batch_size=meta_batch_size, 
                gamma=gamma,
                meta_epsilon=meta_epsilon, 
                cntr_epsilon=cntr_epsilon, 
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
                cntr_lr = cntr_lr
                )

    stats = {
            "train_only_cntr" : init_vars["train_only_cntr"],
            "num_epis" : init_vars["num_epis"],
            "stat_period_dhrl" : init_vars["stat_period_dhrl"],
            "stat_period_cntr" : init_vars["stat_period_cntr"],
            "reset_type" : init_vars["reset_type"], 
            "no_anneal" : init_vars["no_anneal"], 
            "D_in" : D_in, 
            "n_obj":n_obj, 
            "min_num" : min_num, 
            "max_num" : max_num,
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
            "meta_epsilon":meta_epsilon, 
            "cntr_epsilon":cntr_epsilon, 
            "tau" : tau,
            "cntr_memory_size" : cntr_memory_size,
            "meta_memory_size" : meta_memory_size,
            "meta_loss" : meta_loss,
            "meta_optimizer" : meta_optimizer,
            "meta_lr" : meta_lr,
            "cntr_loss" : cntr_loss,
            "cntr_optimizer" : cntr_optimizer,
            "cntr_lr" : cntr_lr,
            'NOTE' : NOTE
            }

    create_directories(init_vars)
    logs_address, models_address = addresses(init_vars)
    with open(logs_address+'stats.txt','w') as f:
        for key, value in stats.items():
            f.write('{0}: {1}\n'.format(key, value))
    with open(models_address+'stats.txt','w') as f:
        for key, value in stats.items():
            f.write('{0}: {1}\n'.format(key, value))
    return env, agent, init_vars

def create_directories(init_vars):
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

    return logs_address, models_address

def train_DHRL(env, agent, init_vars):

    # creating the folders
    logs_address, models_address = addresses(init_vars)
    num_epis = init_vars["num_epis"]

    visits = np.zeros((env.D_in, env.D_in))
    anneal_factor = (1.0-0.1)/(num_epis)
    print("Annealing factor: " + str(anneal_factor))
    game_won_counter = 0
    game_over_counter = 0
    game_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    cntr_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached, non_meta_goal_reached, episode_steps]
    hdqn_logs_list = [] # each slement should be [episode_steps, game_won]
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")
        if init_vars["reset_type"] == 'reset': 
            agent_state, env_state = env.reset() 
        if init_vars["reset_type"] == 'reset_total':
            agent_state, env_state = env.reset_total() 
        visits[agent_state[0,0].item(), agent_state[0,1].item()] += 1
        terminal = False
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.select_goal(agent_state)  # meta cntr selects a goal
            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            total_extr_reward = 0

            agent_env_state_0 = utils.agent_env_state(agent_state, env_state) # for meta controller start state
            meta_goal_reached = False
            cntr_steps = 0
            while not terminal and not meta_goal_reached: # this loop is for meta, while state not terminal
                cntr_steps += 1
                episode_steps += 1
                action_idx, action = agent.select_action(agent_state, env_state, meta_goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")
                next_agent_state, next_env_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                
                extr_reward = torch.tensor([env.extr_reward(next_agent_state)], device=device)
                int_reward = torch.tensor([env.int_reward(next_agent_state, meta_goal)], device=device)
                
                i, j = next_agent_state[0,0].item(), next_agent_state[0,1].item()
                visits[i, j] += 1
                current_element = int(env.grid_mat[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                
                if current_element != meta_goal and current_element != 0:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + 
                #         str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(nextvpn.utexas.edu_agent_state)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    env.update_target_goal()
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


                # if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")
                agent_env_cntr = utils.agent_env_state(agent_state, env_state)
                next_agent_env_cntr = utils.agent_env_state(next_agent_state, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                exp_cntr = copy.deepcopy([agent_env_cntr, action_idx, int_reward,  
                    next_agent_env_cntr, 
                    torch.tensor([meta_goal], dtype=torch.float, device=device), 
                    next_available_actions, cntr_done])
                agent.store(*exp_cntr, meta=False)
                agent.update(meta=False)
                agent.update(meta=True)
                total_extr_reward += extr_reward
                agent_state = next_agent_state
                env_state = next_env_state
                if current_element != 0:
                    cntr_logs_list.append([cntr_steps, meta_goal_reached, current_target_reached, 
                        episode_steps, episode])

            next_agent_env_state = utils.agent_env_state(agent_state, env_state)
            next_available_goals = env.current_objects
            meta_exp_counter += 1 
            exp_meta = copy.deepcopy([agent_env_state_0, torch.tensor([meta_goal], dtype=torch.float, device=device), total_extr_reward, 
                next_agent_env_state, next_available_goals, terminal, meta_exp_counter])
            agent.store(*exp_meta, meta=True)


        hdqn_logs_list.append([episode_steps, game_won, episode])
        print("meta_epsilon: " + str(agent.meta_epsilon))
        print("cntr_epsilon: " + str(agent.cntr_epsilon))

        if episode != 0 and episode % init_vars["stat_period_dhrl"] == 0:

            if not init_vars["no_anneal"]:
                # Annealing 
                agent.meta_epsilon -= anneal_factor * init_vars["stat_period_dhrl"]
                agent.cntr_epsilon -= anneal_factor * init_vars["stat_period_dhrl"]
                # avg_success_rate = cntr_success / goal_selected

                # if(avg_success_rate == 0 or avg_success_rate == 1):
                #     agent.cntr_epsilon -= anneal_factor
                # elif episode > 200:
                #     agent.cntr_epsilon = 1 - avg_success_rate
      
                if(agent.cntr_epsilon < 0.1):
                    agent.cntr_epsilon = 0.1
                # if(agent.meta_epsilon) < 0.05:
                #     agent.meta_epsilon = 0.05


            success_ratio = cntr_success/(cntr_success+cntr_failure)
            cntr_success_stats.append([cntr_success,cntr_failure,goal_selected, success_ratio, 
                batch_episode_counter])
            batch_episode_counter += 1
            cntr_success = 0
            cntr_failure = 0
            goal_selected = 0

            print ("SAVING THE LOG FILES .....")
            with open(logs_address + "cntr_success_stats.txt", "w") as file:
                file.write("cntr_success,cntr_failure,goal_selected, success_ratio, batch_episode_counter \n")
                for r in cntr_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>3} \n'.format(r[0],r[1],r[2],r[3],r[4]))

            with open(logs_address + "game_stats.txt", "w") as file:
                file.write("game_won_counter: {}\n".format(game_won_counter)) 
                file.write("game_over_counter: {}\n".format(game_over_counter))
                file.writet("game won ratio: {}\n".format(game_won_counter/(game_over_counter+game_won_counter)))

            with open(logs_address + "game_result_history.txt", "w") as file:
                file.write("game_result, episode \n")
                for result, ep in game_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))
            # *******************************************************
            with open(logs_address + "hdqn_logs_list.txt", "w") as file:
                file.write("episode_steps, game_won, episode \n")
                for line in hdqn_logs_list:
                    file.write('{0:>5} {1:>6} {2:>6} \n'.format(line[0],line[1],line[2]))
            with open(logs_address + "cntr_logs_list.txt", "w") as file:
                file.write("cntr_steps, meta_goal_reached, current_target_reached, episode_steps, episode \n")
                for line in cntr_logs_list:
                        file.write('{0:>5} {1:>6} {2:>6} {3:>6} {4:>6} \n'.format(line[0],
                            line[1],line[2],line[3],line[4]))



            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_meta_net.state_dict(), models_address + "policy_meta_net")
            torch.save(agent.target_meta_net.state_dict(), models_address + "target_meta_net")
            torch.save(agent.policy_cntr_net.state_dict(), models_address + "policy_cntr_net")
            torch.save(agent.target_cntr_net.state_dict(), models_address + "target_cntr_net")

def train_cntr(env, agent, init_vars):

    # creating the folders
    logs_address, models_address = addresses(init_vars)
    num_epis = init_vars["num_epis"]

    visits = np.zeros((env.D_in, env.D_in))
    anneal_factor = (1.0-0.1)/(num_epis)
    print("Annealing factor: " + str(anneal_factor))
    game_won_counter = 0
    game_over_counter = 0
    cntr_result_history = []
    meta_exp_counter = 0
    goal_selected = 0
    cntr_success = 0
    cntr_failure = 0
    cntr_success_stats = []
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached]
    batch_episode_counter = 0
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")

        if init_vars["reset_type"] == 'reset': 
            agent_state, env_state = env.reset() 
        if init_vars["reset_type"] == 'reset_total':
            agent_state, env_state = env.reset_total() 

        visits[agent_state[0,0].item(), agent_state[0,1].item()] += 1
        terminal = False
        non_meta_goal_reached = False
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.random_goal_selection()  # meta cntr selects a goal
            goal_selected += 1
            print("\nNew Goal: "  + str(meta_goal) + "\n")
            meta_goal_reached = False
            cntr_steps = 0
            while not terminal and not meta_goal_reached: # this loop is for meta, while state not terminal
                cntr_steps += 1
                episode_steps += 1
                action_idx, action = agent.select_action(agent_state, env_state, meta_goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")
                next_agent_state, next_env_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                int_reward = env.int_reward(next_agent_state, meta_goal)
                
                i, j = next_agent_state[0,0].item(), next_agent_state[0,1].item()
                visits[i, j] += 1
                current_element = int(env.grid_mat[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                if current_element != meta_goal and current_element != 0:
                    non_meta_goal_reached = True
                else:
                    non_meta_goal_reached = False

                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + 
                #         str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_state)
                terminal = game_over or game_won # terminal refers to next state
                cntr_done = non_meta_goal_reached

                if current_target_reached:
                    # print("CURRENT TARGET REACHED! ")
                    # print("the object reached : " + str(current_element))
                    # print("the current target : " + str(env.current_target_goal))
                    # print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    env.update_target_goal()
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


                # if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")

                agent_env_state = utils.agent_env_state(agent_state, env_state)
                next_agent_env_state = utils.agent_env_state(next_agent_state, next_env_state)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                exp_cntr = copy.deepcopy([agent_env_state, action_idx, int_reward,  
                    next_agent_env_state, torch.tensor([meta_goal], dtype=torch.float, device=device), 
                    next_available_actions, cntr_done])
                agent.store(*exp_cntr, meta=False)
                agent.update(meta=False)
                # agent.update(meta=True)

                agent_state = next_agent_state
                env_state = next_env_state

                if current_element != 0:
                    cntr_logs_list.append([cntr_steps, meta_goal_reached, current_target_reached, 
                        episode_steps, episode])



        # hdqn_logs_list.append([episode_steps, game_won])
        # print("meta_epsilon: " + str(agent.meta_epsilon))
        print("cntr_epsilon: " + str(agent.cntr_epsilon))

        if episode != 0 and episode % init_vars["stat_period_cntr"] == 0:

            if not init_vars["no_anneal"]:
                # Annealing 
                agent.cntr_epsilon -= anneal_factor * init_vars["stat_period_cntr"]
                # if episode < num_epis / 10:
                #     agent.cntr_epsilon -= anneal_factor * init_vars["stat_period_cntr"]
                # else:
                #     success_ratio = cntr_success / (cntr_success + cntr_failure)
                #     if success_ratio == 1 or success_ratio == 0:
                #         agent.cntr_epsilon -= anneal_factor
                #     else:
                #         agent.cntr_epsilon = 1 - success_ratio
                    
                if agent.cntr_epsilon < 0.1:
                    agent.cntr_epsilon = 0.1


            success_ratio = cntr_success/(cntr_success+cntr_failure)
            cntr_success_stats.append([cntr_success,cntr_failure,goal_selected, success_ratio, 
                batch_episode_counter])
            batch_episode_counter += 1
            cntr_success = 0
            cntr_failure = 0
            goal_selected = 0

            with open(logs_address + "cntr_success_stats.txt", "w") as file:
                file.write("cntr_success,cntr_failure,goal_selected, success_ratio, batch_episode_counter \n")
                for r in cntr_success_stats:
                    file.write('{0:>7} {1:>7} {2:>7} {3:>6.3f} {4:>3} \n'.format(r[0],r[1],r[2],r[3],r[4]))

            with open(logs_address + "cntr_result_history.txt", "w") as file:
                file.write("cntr_result, episode \n")
                for result, ep in cntr_result_history:
                    file.write('{0:>5}  {1:>9} \n'.format(result, ep))


            with open(logs_address + "cntr_logs_list.txt", "w") as file:
                file.write("cntr_steps, meta_goal_reached, current_target_reached, episode_steps, episode \n")
                for line in cntr_logs_list:
                        file.write('{0:>5} {1:>6} {2:>6} {3:>6} {4:>6} \n'.format(line[0],
                            line[1],line[2],line[3],line[4]))



            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_cntr_net.state_dict(), models_address + "policy_cntr_net_train_cntr")
            torch.save(agent.target_cntr_net.state_dict(), models_address + "target_cntr_net_train_cntr")

def main(args):

    env, agent, init_vars = init(args)

    if init_vars["train_only_cntr"] :
        train_cntr(env, agent, init_vars)
    else:
        train_DHRL(env, agent, init_vars)
    
            
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Relational-RL')
    parser.add_argument('--fileName',type=str, default="0")
    parser.add_argument('--train_only_cntr ', default=False)

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