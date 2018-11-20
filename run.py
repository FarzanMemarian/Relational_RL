# using some of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from envs.gridworld_relational import Gridworld
from agent.agent import hDQN
from pdb import set_trace 

plt.style.use('ggplot')

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():

    ActorExperience = namedtuple("ActorExperience", 
        ["agent_state", "goal", "action", "reward", "next_agent_state", "done"])
    MetaExperience = namedtuple("MetaExperience",   
        ["agent_state", "goal", "reward", "next_agent_state", "done"])
    # properties of the grid-world
    n_dim = 5
    start = np.asarray([0,1])
    n_obj = 4
    min_num = 1
    max_num = 30

    # REWARDS
    not_moving_reward = -1
    terminal_reward = -10000
    step_reward = -1
    goal_reward = 100
    final_goal_reward = 10000
    intrinsic_goal_reward = 100
    intrinsic_step_reward = -1
    intrinsic_wrong_goal_reward = -10000

    env = Gridworld(n_dim, start, n_obj, min_num, max_num,
        not_moving_reward, terminal_reward, step_reward, goal_reward, final_goal_reward,
        intrinsic_goal_reward, intrinsic_step_reward, intrinsic_wrong_goal_reward)

    # PARAMETERS OF ENVIRONMENT
    meta_input_dim = 2 + n_dim**2
    input_dim = 2 + 1 + n_dim**2
    meta_nodes = [20, 30, 30, 30, n_obj] 
    nodes = [20, 30, 30, 30, 4] 
    actor_epsilon = 1.0
    meta_epsilon = 1.0

    agent = hDQN(env=env, meta_input_dim=meta_input_dim, input_dim=input_dim, 
        meta_nodes=meta_nodes, nodes=nodes, actor_epsilon=actor_epsilon, 
        meta_epsilon=meta_epsilon)


    num_thousands = 12
    num_epis = 100
    visits = np.zeros((num_thousands, n_dim, n_dim))
    anneal_factor = (1.0-0.1)/(num_thousands * num_epis)
    print("Annealing factor: " + str(anneal_factor))
    for episode_thousand in range(num_thousands):
        for episode in range(num_epis):
            print("\n\n### EPISODE "  + str(episode_thousand*num_thousands + episode) + "###")
            agent_state = env.reset() # the returned agent_state is just a (2,) numpy array 
            visits[episode_thousand, agent_state[0], agent_state[1]] += 1

            
            
            while not env.is_terminal(agent_state):  # this loop is for meta-controller, while state not terminal
                goal_idx, goal = agent.select_goal(agent_state)  # meta controller selects a goal
                agent.goal_selected[goal_idx] += 1
                print("\nNew Goal: "  + str(goal) + "\nState-Actions: ")
                total_extrinsic_reward = 0
                s0_agent = agent_state
                selected_goal_reached = False
                while not env.is_terminal(agent_state) and not selected_goal_reached: # this loop is for meta, while state not terminal
                    action_idx, action = agent.select_action(agent_state, goal, goal_idx) # controller selects an action among permitable actions
                    # print(str((state,action)) + "; ")
                    extrinsic_reward, next_agent_state = env.take_action(action_idx) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                    print ("action ----> "  + action)
                    print ("state after action ----> " + "[" + str(next_agent_state[0]) +", " + str(next_agent_state[1]) + "]" )
                    print ("---------------------")
                    visits[episode_thousand, next_agent_state[0], next_agent_state[1]] += 1
                    intrinsic_reward, selected_goal_reached = env.intrinsic_critique(next_agent_state, goal)
                    if selected_goal_reached:
                        agent.goal_success[goal_idx] += 1
                        print("selected goal reached!! ")
                    if env.is_terminal(next_agent_state):
                        print("terminal state, the object reached is: " +  str(env.grid_mat[next_agent_state[0], next_agent_state[1]]))
                    # if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                    #     print("final object/number picked!! ")
                    exp = ActorExperience(agent_state, goal, action_idx, intrinsic_reward, next_agent_state, env.is_terminal(next_agent_state))
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_extrinsic_reward += extrinsic_reward
                    agent_state = next_agent_state
                exp = MetaExperience(s0_agent, goal, total_extrinsic_reward, agent_state, env.is_terminal(agent_state))
                agent.store(exp, meta=True)
                set_trace()

            #Annealing 
            agent.meta_epsilon -= anneal_factor
            avg_success_rate = agent.goal_success[goal_idx] / agent.goal_selected[goal_idx]
            agent.actor_epsilon -= anneal_factor

            # if(avg_success_rate == 0 or avg_success_rate == 1):
            #     agent.actor_epsilon -= anneal_factor
            # else:
            #     agent.actor_epsilon = 1- avg_success_rate
        
            # if(agent.actor_epsilon < 0.1):
            #     agent.actor_epsilon = 0.1
            print("meta_epsilon: " + str(agent.meta_epsilon))
            print("actor_epsilon: " + str(agent.actor_epsilon))
            
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