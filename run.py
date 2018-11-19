# starting from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

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
    ActorExperience = namedtuple("ActorExperience", ["agent_state", "goal", "action", "reward", "next_agent_state", "done"])
    MetaExperience = namedtuple("MetaExperience",   ["agent_state", "goal", "reward", "next_agent_state", "done"])
    # properties of the grid-world
    n_dim = 9
    start = np.asarray([0,1])
    n_obj = 4
    min_num = 1
    max_num = 20
    env = Gridworld(n_dim, start, n_obj, min_num, max_num)
    agent = hDQN(env=env)


    num_thousands = 12
    num_epis = 1000
    visits = np.zeros((num_thousands, n_dim, n_dim))
    anneal_factor = (1.0-0.1)/(num_thousands * num_epis)
    print("Annealing factor: " + str(anneal_factor))
    for episode_thousand in range(num_thousands):
        for episode in range(num_epis):
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            agent_state = env.reset() # the returned agent_state is just a 1*2 numpy array 
            visits[episode_thousand, agent_state[0], agent_state[1]] += 1
            done = False
            while not done:  # this loop is for meta-controller, while state not terminal
                goal_idx, goal = agent.select_goal(agent_state)  # meta controller selects a goal
                agent.goal_selected[goal_idx] += 1
                print("\nNew Goal: "  + str(goal) + "\nState-Actions: ")
                total_external_reward = 0
                goal_reached = False
                s0_agent = agent_state
                while not done and not goal_reached: # this loop is for meta, while state not terminal
                    action = agent.select_move(agent_state, goal, goal_idx) # controller selects an action among permitable actions
                    # print(str((state,action)) + "; ")
                    external_reward, next_agent_state, done = env.take_action(action) # RIGHT NOW THE DONE IS NOT IMPLEMENTED YET 
                    visits[episode_thousand, next_agent_state[0], next_agent_state[1]] += 1
                    intrinsic_reward = env.intrinsic_reward(next_agent_state, goal)
                    goal_reached = next_state == goal
                    if goal_reached:
                        agent.goal_success[goal_idx] += 1
                        print("Goal reached!! ")
                    if env.grid_mat[next_agent_state[0], next_agent_state[1]] == env.original_objects[-1]:
                        print("final object/number picked!! ")
                    exp = ActorExperience(agent_state, goal, action, intrinsic_reward, next_agent_state, done)
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_external_reward += external_reward
                    agent_state = next_agent_state
                exp = MetaExperience(s0_agent, goal, total_external_reward, next_agent_state, done)
                agent.store(exp, meta=True)
                
                #Annealing 
                agent.meta_epsilon -= anneal_factor
                avg_success_rate = agent.goal_success[goal-1] / agent.goal_selected[goal-1]
                
                if(avg_success_rate == 0 or avg_success_rate == 1):
                    agent.actor_epsilon[goal-1] -= anneal_factor
                else:
                    agent.actor_epsilon[goal-1] = 1- avg_success_rate
            
                if(agent.actor_epsilon[goal-1] < 0.1):
                    agent.actor_epsilon[goal-1] = 0.1
                print("meta_epsilon: " + str(agent.meta_epsilon))
                print("actor_epsilon " + str(goal) + ": " + str(agent.actor_epsilon[goal-1]))
                
            if (episode % 100 == 99):
                print("")
                print(str(visits/1000) + "")

    eps = list(range(1,13))
    plt.subplot(2, 3, 1)
    plt.plot(eps, visits[:,0]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S1")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, visits[:,1]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S2")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, visits[:,2]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S3")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, visits[:,3]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S4")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, visits[:,4]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S5")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, visits[:,5]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S6")
    plt.grid(True)
    plt.savefig('first_run.png')
    plt.show()

if __name__ == "__main__":
    main()