# using some of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import sys
sys.path.append('../')
from envs.gridworld2 import Gridworld
from agent.agent2 import hDQN
from utils import utils
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pdb import set_trace 
import pickle

plt.style.use('ggplot')

def init():
    # TRAIN HRL PARAMS
    num_epis = 1000

    # GRID WORLD GEOMETRICAL PARAMETERS
    D_in = 5 # pick odd numbers
    start = torch.zeros([1,2], dtype=torch.int)
    start[0,0] = 0
    start[0,1] = 1
    n_obj = 2
    min_num = 1
    max_num = 30

    # extr REWARDS
    not_moving_reward = -1
    game_over_reward = -100
    step_reward = -1
    current_goal_reward = 10
    final_goal_reward = 100

    # int REWARDS
    int_goal_reward = 10
    int_step_reward = -1
    int_wrong_goal_reward = -20

    # PARAMETERS OF MEATA cntr
    meta_batch_size = 30
    meta_epsilon = 1
    meta_memory_size = 10000

    # PARAMETERS OF THE cntr
    batch_size = 30
    gamma = 0.975
    epsilon = 1
    tau = 0.001
    cntr_memory_size = 10000


    cntr_Transition = namedtuple("cntr_Transition", 
        ["agent_env_goal_cntr", "action_idx", "int_reward", "next_agent_env_goal_cntr", "goal", 
        "next_available_actions", "cntr_done"])
    meta_Transition = namedtuple("meta_Transition",   
        ["agent_env_state_0", "goal", "reward", "next_agent_env_state", 
        "next_available_goals", "terminal", "meta_exp_counter"])

    # create and initialize the environment   
    env = Gridworld(D_in, 
                    start, 
                    n_obj, 
                    min_num, 
                    max_num,
                    not_moving_reward, 
                    game_over_reward, 
                    step_reward, 
                    current_goal_reward, 
                    final_goal_reward,
                    int_goal_reward, 
                    int_step_reward, 
                    int_wrong_goal_reward)

    # create and initialize the agent
    agent = hDQN(env=env, 
                batch_size=batch_size,
                meta_batch_size=meta_batch_size, 
                gamma=gamma,
                meta_epsilon=meta_epsilon, 
                epsilon=epsilon, 
                tau = tau,
                cntr_Transition = cntr_Transition,
                cntr_memory_size = cntr_memory_size,
                meta_Transition = meta_Transition,
                meta_memory_size = meta_memory_size)

    return env, agent, num_epis

def train_HRL(env, agent, num_epis=10):


    visits = np.zeros((env.D_in, env.D_in))
    anneal_factor = (1.0-0.1)/(num_epis)
    print("Annealing factor: " + str(anneal_factor))
    game_won_counter = 0
    game_over_counter = 0
    game_result_history = []
    meta_exp_counter = 0
    cntr_logs_list = [] # each element should be [cntr_steps, meta_goal_reached]
    hdqn_logs_list = [] # each slement should be [episode_steps, game_won]
    for episode in range(num_epis):
        print("\n\n\n\n### EPISODE "  + str(episode) + "###")
        agent_state, env_state = env.reset() 
        visits[agent_state[0,0].item(), agent_state[0,1].item()] += 1
        terminal = False
        episode_steps = 0
        while not terminal:  # this loop is for meta-cntr
                             # which means while the game is not lost or won
            meta_goal = agent.select_goal(agent_state)  # meta cntr selects a goal
            # agent.goal_selected[goal_idx] += 1
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
                extr_reward = env.extr_reward(next_agent_state)
                int_reward = env.int_reward(next_agent_state, meta_goal)
                
                i, j = next_agent_state[0,0].item(), next_agent_state[0,1].item()
                visits[i, j] += 1
                current_element = int(env.grid_mat[i,j].item())
                meta_goal_reached = current_element == meta_goal
                current_target_reached = current_element == env.current_target_goal
                # print ("action ----> "  + action)
                # print ("state after action ----> " + "[" + str(i) +", " + 
                #         str(j) + "]" )
                # print ("---------------------")
                game_over, game_won = env.is_terminal(next_agent_state)
                terminal = game_over or game_won # terminal refers to next state

                if current_target_reached:
                    print("CURRENT TARGET REACHED! ")
                    print("the object reached : " + str(current_element))
                    print("the current target : " + str(env.current_target_goal))
                    print ("original objects: {}".format(env.original_objects))
                    removed_object = env.remove_object(i,j)
                    env.update_target_goal()
                    # object will only be removed if it's the right object to be picked up
                    # else the game ends and it doesn't matter what the remaining objects are
                    print ("object {} removed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(removed_object))     
                    print ("current objects: {}".format(env.current_objects))
                    print ("********************") 
                
                if meta_goal_reached:
                    # agent.goal_success[goal_idx] += 1
                    print("SELECTED GOAL REACHED! ")
                    print("the object reached : " + str(current_element))
                    print("the meta goal : " + str(meta_goal))
                    print ("original objects: {}".format(env.original_objects))
                    print ("current objects: {}".format(env.current_objects))
                    print ("********************")

                if game_over:
                    game_over_counter += 1
                    game_result_history.append(0)
                    print("GAME OVER!!!") 
                    print("selected goal:" + str(meta_goal))
                    print("the object reached: " +  str(current_element))
                    print("the current target goal: " + str(env.current_target_goal))
                    print ("original objects: {}".format(env.original_objects))
                    print ("current objects: {}".format(env.current_objects))                        
                    print("********************")
                
                if game_won:
                    game_won_counter += 1
                    game_result_history.append(1)
                    print("GAME WON!!!") 
                    print("the object reached: " +  str(current_element))
                    print("the current target goal: " + str(env.current_target_goal))
                    print ("original objects: {}".format(env.original_objects)) 
                    print ("current objects: {}".format(env.current_objects))                       
                    print("********************")


                # if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                #     print("final object/number picked!! ")
                cntr_done = meta_goal_reached or terminal # note that cntr_done refers 
                                                                    # to next state
                agent_env_goal_cntr = utils.cntr_input(env.D_in, agent_state, env_state, meta_goal)
                next_agent_env_goal_cntr = utils.cntr_input(env.D_in, next_agent_state, next_env_state, meta_goal)
                next_available_actions = env.allowable_actions[i,j]

                # if the state is terminal, the experiment will not even be added to the cntr_Transition
                # or the MetaEcperience, only the next_state can be terminal
                exp_cntr = copy.deepcopy([agent_env_goal_cntr, action_idx, int_reward,  
                    next_agent_env_goal_cntr, meta_goal, next_available_actions, cntr_done])
                # agent.store(*exp_cntr, meta=False)
                agent.update(meta=False)
                agent.update(meta=True)
                total_extr_reward += extr_reward
                agent_state = next_agent_state
                env_state = next_env_state

            cntr_logs_list.append([cntr_steps, meta_goal_reached])

            next_agent_env_state = utils.agent_env_state(agent_state, env_state)
            next_available_goals = env.current_objects

            # print ("next available goals {} *******************************".format(
            #     next_available_goals))
            # print ("terminal?    {}".format(terminal))
            meta_exp_counter += 1 
            exp_meta = copy.deepcopy([agent_env_state_0, meta_goal, total_extr_reward, next_agent_env_state, 
                next_available_goals, terminal, meta_exp_counter])
            # if not terminal and len(next_available_goals) == 1:
            #     print ("next available goals {} *******************************".format(
            #     next_available_goals))
            #     print ("terminal?    {}".format(terminal))
            #     set_trace()
            agent.store(*exp_meta, meta=True)
            # set_trace()

            # Annealing 
            agent.meta_epsilon -= anneal_factor
            agent.epsilon -= anneal_factor
            # avg_success_rate = agent.goal_success[goal_idx] / agent.goal_selected[goal_idx]

            # if(avg_success_rate == 0 or avg_success_rate == 1):
            #     agent.epsilon -= anneal_factor
            # else:
            #     agent.epsilon = 1 - avg_success_rate
        
            if(agent.epsilon < 0.1):
                agent.epsilon = 0.1
            if(agent.meta_epsilon) < 0.1:
                agent.meta_epsilon = 0.1


        hdqn_logs_list.append([episode_steps, game_won])
        print("meta_epsilon: " + str(agent.meta_epsilon))
        print("epsilon: " + str(agent.epsilon))


        if episode >= 100 and episode % 100 == 0:
            print("SAVING THE LOG FILES .........")
            with open("../logs/logs.txt", "w") as file:
                file.write("game_won_counter: {}\n".format(game_won_counter)) 
                file.write("game_over_counter: {}\n".format(game_over_counter))

            with open("../logs/game_result_history.txt", "w") as file:
                for game in game_result_history:
                    # item = game + "\n"
                    item = str(game) + "\n"
                    file.write(item)

            with open("../logs/game_result_history.pickle","wb") as file:
                pickle.dump(game_result_history, file)

            # *******************************************************
            with open("../logs/hdqn_logs_list.txt", "w") as file:
                for line in hdqn_logs_list:
                    file.write(str(line[0]) + "   " +  str(line[1]) + "\n")
            with open("../logs/cntr_logs_list.txt", "w") as file:
                for line in cntr_logs_list:
                    file.write(str(line[0]) + "   " +  str(line[1]) + "\n")

            with open("../logs/hdqn_logs_list.pickle", "wb") as file:
                pickle.dump(hdqn_logs_list, file)
            with open("../logs/cntr_logs_list.pickle", "wb") as file:
                pickle.dump(cntr_logs_list, file)


            print ("SAVING THE MODELS .............")  
            print ()
            torch.save(agent.policy_meta_net.state_dict(), "../saved_models/policy_meta_net")
            torch.save(agent.target_meta_net.state_dict(), "../saved_models/target_meta_net")
            torch.save(agent.policy_cntr_net.state_dict(), "../saved_models/policy_cntr_net")
            torch.save(agent.target_cntr_net.state_dict(), "../saved_models/target_cntr_net")
            # # serialize model to JSON
            # model_json = agent.policy_meta_net.to_json()
            # with open("../saved_models/policy_meta_net.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # agent.policy_meta_net.save_weights("../saved_models/policy_meta_net.h5")
            # print("Saved model to disk")

            # # serialize model to JSON
            # model_json = agent.target_meta_net.to_json()
            # with open("../saved_models/target_meta_net.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # agent.target_meta_net.save_weights("../saved_models/target_meta_net.h5")
            # print("Saved model to disk")

            # # serialize model to JSON
            # model_json = agent.policy_cntr_net.to_json()
            # with open("../saved_models/policy_cntr_net.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # agent.policy_cntr_net.save_weights("../saved_models/policy_cntr_net.h5")
            # print("Saved model to disk")

            # model_json = agent.target_cntr_net.to_json()
            # with open("../saved_models/target_cntr_net.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # agent.target_cntr_net.save_weights("../saved_models/target_cntr_net.h5")
            # print("Saved model to disk")


def train_cntr(env, agent):

    cntr_Transition = namedtuple("cntr_Transition", 
        ["agent_state", "goal", "action", "reward", "next_agent_state", "done"])
    num_epis = 1000
    anneal_factor = (1.0-0.1)/(num_epis)
    print("Annealing factor: " + str(anneal_factor))
    for episode in range(num_epis):
        print("\n\n### EPISODE "  + str(episode) + "###")
        agent_state = env.reset() # the returned agent_state is just a (2,) numpy array 

        terminal = False
        while not terminal:  # this loop is for meta-cntr, while state not terminal
            goal = agent.select_goal(agent_state)  # meta cntr selects a goal
            # agent.goal_selected[goal_idx] += 1
            print("\nNew Goal: "  + str(goal) + "\nState-Actions: ")
            s0_agent = agent_state
            meta_goal_reached = False
            while not terminal and not meta_goal_reached:
                action_idx, action = agent.select_action(agent_state, goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")
                extr_reward, next_agent_state = env.step(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                int_reward, meta_goal_reached = env.int_reward(next_agent_state, meta_goal)


                print ("action ----> "  + action)
                print ("state after action ----> " + "[" + str(next_agent_state[0,0]) +", " + 
                        str(next_agent_state[0,1]) + "]" )
                print ("---------------------")

                terminal = env.is_terminal(next_agent_state)[0] or env.is_terminal(next_agent_state)[1]
                object_reached = env.grid_mat[next_agent_state[0,0], next_agent_state[0,1]]
                if meta_goal_reached:
                    # agent.goal_success[goal_idx] += 1
                    print("SELECTED GOAL REACHED! ")
                    print("the object reached which should equal the selected goal: " + str(object_reached))
                    print ("original objects: {}".format(env.original_objects))
                    print ("********************")
                if env.is_terminal(next_agent_state)[0]:
                    print("GAME OVER!!!") 
                    print("selected goal:" + str(goal))
                    print("the object reached: " +  str(object_reached))
                    print("the current target goal: " + str(env.current_target_goal))
                    print ("original objects: {}".format(env.original_objects))                        
                    print("********************")
                if env.is_terminal(next_agent_state)[1]:
                    print("GAME WON!!!") 
                    print("the object reached: " +  str(object_reached))
                    print("the current target goal: " + str(env.current_target_goal))
                    print ("original objects: {}".format(env.original_objects))                        
                    print("********************")
                
                exp = cntr_Transition(agent_state, goal, action_idx, int_reward, 
                                            next_agent_state, meta_goal_reached)
                agent.store(exp, meta=False)
                agent.update(meta=False)
                agent_state = next_agent_state
        

        #Annealing, just anneal the cntr epsilon
        # avg_success_rate = agent.goal_success[goal_idx] / agent.goal_selected[goal_idx]
        annealgent.epsilon -= anneal_factor
        
        # if(avg_success_rate == 0 or avg_success_rate == 1):
        #     agent.epsilon -= anneal_factor
        # else:
        #     agent.epsilon = 1- avg_success_rate
    
        # if(agent.epsilon < 0.1):
        #     agent.epsilon = 0.1
        print("epsilon: " + str(agent.epsilon))

    
    # SAVING MODELS AND THEIR WEIGHTS
    # serialize model to JSON
    model_json = agent.cntr.to_json()
    with open("../saved_models/cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.cntr.save_weights("../saved_models/cntr.h5")
    print("Saved model to disk")

    # serialize model to JSON
    model_json = agent.target_cntr.to_json()
    with open("../saved_models/target_cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.meta_cntr.save_weights("../saved_models/target_cntr.h5")
    print("Saved model to disk")



def main():

    env, agent, num_epis = init()
    train_HRL(env, agent, num_epis)
    # set_trace()
    # train_cntr(env, agent)
    
            
    # eps = list(range(1,13))
    # plt.subplot(2, 3, 1)
    # plt.plot(eps, visits[:,0]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S1")
    # plt.grid(True)

    # plt.subplot(2, 3, 2)
    # plt.plot(eps, visits[:,1]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S2")
    # plt.grid(True)

    # plt.subplot(2, 3, 3)
    # plt.plot(eps, visits[:,2]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S3")
    # plt.grid(True)

    # plt.subplot(2, 3, 4)
    # plt.plot(eps, visits[:,3]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S4")
    # plt.grid(True)

    # plt.subplot(2, 3, 5)
    # plt.plot(eps, visits[:,4]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S5")
    # plt.grid(True)

    # plt.subplot(2, 3, 6)
    # plt.plot(eps, visits[:,5]/1000)
    # plt.xlabel("Episodes (*1000)")
    # plt.ylim(-0.01, 2.0)
    # plt.xlim(1, 12)
    # plt.title("S6")
    # plt.grid(True)
    # plt.savefig('first_run.png')
    # plt.show()

if __name__ == "__main__":
    main()