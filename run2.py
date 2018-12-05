# using some of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from envs.gridworld2 import Gridworld
from agent.agent2 import hDQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pdb import set_trace 
import pickle

plt.style.use('ggplot')

def init():
    # TRAIN HRL PARAMS
    num_thousands = 1
    num_epis = 100

    # GRID WORLD GEOMETRICAL PARAMETERS
    n_dim = 3 # pick odd numbers
    start = torch.zeros([1,2], dtype=torch.int)
    start[0,0] = 0
    start[0,1] = 1
    n_obj = 2
    min_num = 1
    max_num = 30

    # EXTRINSIC REWARDS
    not_moving_reward = -1
    game_over_reward = -1000
    step_reward = -1
    current_goal_reward = 300
    final_goal_reward = 10000

    # int REWARDS
    int_goal_reward = 200
    int_step_reward = -1
    int_wrong_goal_reward = -200

    # PARAMETERS OF MEATA cntr
    meta_batch_size = 1000
    meta_epsilon = 1
    meta_memory_size = 10000

    # PARAMETERS OF THE cntr
    batch_size = 100
    gamma = 0.975
    epsilon = 1
    tau = 0.001
    memory_size = 10000

    # create and initialize the environment   
    env = Gridworld(n_dim, 
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
                memory_size = memory_size,
                meta_memory_size = meta_memory_size)

    return env, agent, num_thousands, num_epis

def train_HRL(env, agent, num_thousands=12, num_epis=10):

    cntrExperience = namedtuple("cntrExperience", 
        ["state", "goal", "action", "reward", "next_state", "next_available_actions","cntr_done"])
    MetaExperience = namedtuple("MetaExperience",   
        ["state", "goal", "reward", "next_state", "next_available_goals", "done"])

    visits = np.zeros((num_thousands, env.n_dim, env.n_dim))
    anneal_factor = (1.0-0.1)/(num_thousands * num_epis)
    print("Annealing factor: " + str(anneal_factor))
    game_won_counter = 0
    game_over_counter = 0
    game_result_history = []
    for episode_thousand in range(num_thousands):
        for episode in range(num_epis):
            print("\n\n\n\n### EPISODE "  + str(episode_thousand*num_thousands + episode) + "###")
            agent_state, env_state = env.reset() # the returned agent_state is just a (2,) numpy array 
            set_trace()
            visits[episode_thousand, agent_state[0,0], agent_state[0,1]] += 1
            terminal = False
            while not terminal:  # this loop is for meta-cntr, while state not terminal
                goal = agent.select_goal(agent_state)  # meta cntr selects a goal
                # agent.goal_selected[goal_idx] += 1
                print("\nNew Goal: "  + str(goal) + "\nState-Actions: ")
                total_extrinsic_reward = 0
                s0 = state = np.concatenate((agent_state, env_state_flat), axis=1)
                selected_goal_reached = False
                while not terminal and not selected_goal_reached: # this loop is for meta, while state not terminal
                    action_idx, action = agent.select_action(agent_state, goal) # cntr selects an action among permitable actions
                    # print(str((state,action)) + "; ")
                    extrinsic_reward, next_agent_state, next_env_state_flat = env.take_action(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                    i, j = next_agent_state[0,0], next_agent_state[0,1]
                    visits[episode_thousand, i, j] += 1
                    int_reward, selected_goal_reached = env.int_critique(next_agent_state, goal)

                    # print ("action ----> "  + action)
                    # print ("state after action ----> " + "[" + str(i) +", " + 
                    #         str(j) + "]" )
                    # print ("---------------------")

                    game_over = env.is_terminal(next_agent_state)[0]
                    game_won = env.is_terminal(next_agent_state)[1]
                    terminal = game_over or game_won # terminal refers to next state 
                    object_reached = env.grid_mat[i,j]
                    if selected_goal_reached:
                        # agent.goal_success[goal_idx] += 1
                        print("SELECTED GOAL REACHED! ")
                        print("the object reached which should equal the selected goal: " + str(object_reached))
                        print ("original objects: {}".format(env.original_objects))
                        if object_reached == env.current_target_goal:
                            env.remove_occupied_object()
                        print ("current objects: {}".format(env.current_objects))
                        print ("********************")

                    if game_over:
                        game_over_counter += 1
                        game_result_history.append(0)
                        print("GAME OVER!!!") 
                        print("selected goal:" + str(goal))
                        print("the object reached: " +  str(object_reached))
                        print("the current target goal: " + str(env.current_target_goal))
                        print ("original objects: {}".format(env.original_objects))
                        print ("current objects: {}".format(env.current_objects))                        
                        print("********************")
                    if game_won:
                        game_won_counter += 1
                        game_result_history.append(1)
                        print("GAME WON!!!") 
                        print("the object reached: " +  str(object_reached))
                        print("the current target goal: " + str(env.current_target_goal))
                        print ("original objects: {}".format(env.original_objects)) 
                        print ("current objects: {}".format(env.current_objects))                       
                        print("********************")


                    # if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                    #     print("final object/number picked!! ")
                    cntr_done = selected_goal_reached or terminal # note that cntr_done refers 
                                                                        # to next stat

                    
                    state = np.concatenate((env_state_flat, agent_state), axis=1)
                    next_state = np.concatenate((next_env_state_flat, next_agent_state), axis=1)
                    next_available_goals = env.current_objects
                    next_available_actions = env.allowable_actions[i,j]

                    # if the state is terminal, the experiment will not even be added to the cntrExperience
                    # or the MetaEcperience, only the next_state can be terminal
                    exp = cntrExperience(state, goal.reshape((1,1)), action_idx, int_reward,  
                        next_state, next_available_actions, cntr_done)
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_extrinsic_reward += extrinsic_reward
                    agent_state = next_agent_state
                exp = MetaExperience(s0, goal.reshape((1,1)), total_extrinsic_reward, next_state, 
                    next_available_goals, terminal)
                agent.store(exp, meta=True)
                # set_trace()

                #Annealing 
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
                    agent.epsilon = 0.1


            print("meta_epsilon: " + str(agent.meta_epsilon))
            print("epsilon: " + str(agent.epsilon))

    print("SAVING THE LOG FILES .........")
    with open("logs/logs.txt", "w") as file:
        file.write("game_won_counter: {}\n".format(game_won_counter)) 
        file.write("game_over_counter: {}\n".format(game_over_counter))

    with open("logs/game_result_history.txt", "w") as file:
        for game in game_result_history:
            # item = game + "\n"
            item = str(game) + "\n"
            file.write(item)

    with open("logs/game_result_history.pickle","wb") as file:
        pickle.dump(game_result_history, file)


    print ("SAVING THE MODELS .............")  
    print ()
    # serialize model to JSON
    model_json = agent.meta_cntr.to_json()
    with open("saved_models/meta_cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.meta_cntr.save_weights("saved_models/meta_cntr.h5")
    print("Saved model to disk")

    # serialize model to JSON
    model_json = agent.target_meta_cntr.to_json()
    with open("saved_models/target_meta_cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.target_meta_cntr.save_weights("saved_models/target_meta_cntr.h5")
    print("Saved model to disk")

    # serialize model to JSON
    model_json = agent.cntr.to_json()
    with open("saved_models/cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.cntr.save_weights("saved_models/cntr.h5")
    print("Saved model to disk")

  
    model_json = agent.target_cntr.to_json()
    with open("saved_models/target_cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.meta_cntr.save_weights("saved_models/target_cntr.h5")
    print("Saved model to disk")


def train_cntr(env, agent):

    cntrExperience = namedtuple("cntrExperience", 
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
            selected_goal_reached = False
            while not terminal and not selected_goal_reached:
                action_idx, action = agent.select_action(agent_state, goal) # cntr selects an action among permitable actions
                # print(str((state,action)) + "; ")
                extrinsic_reward, next_agent_state = env.take_action(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                int_reward, selected_goal_reached = env.int_critique(next_agent_state, goal)


                print ("action ----> "  + action)
                print ("state after action ----> " + "[" + str(next_agent_state[0,0]) +", " + 
                        str(next_agent_state[0,1]) + "]" )
                print ("---------------------")

                terminal = env.is_terminal(next_agent_state)[0] or env.is_terminal(next_agent_state)[1]
                object_reached = env.grid_mat[next_agent_state[0,0], next_agent_state[0,1]]
                if selected_goal_reached:
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
                
                exp = cntrExperience(agent_state, goal, action_idx, int_reward, 
                                            next_agent_state, selected_goal_reached)
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
    with open("saved_models/cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.cntr.save_weights("saved_models/cntr.h5")
    print("Saved model to disk")

    # serialize model to JSON
    model_json = agent.target_cntr.to_json()
    with open("saved_models/target_cntr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.meta_cntr.save_weights("saved_models/target_cntr.h5")
    print("Saved model to disk")



def main():

    env, agent, num_thousands, num_epis = init()
    train_HRL(env, agent, num_thousands, num_epis)
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