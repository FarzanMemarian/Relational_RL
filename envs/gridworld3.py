import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pdb import set_trace


""" NOTES
- in steps, right now if the action is illegal, ncdothing happens and no reward is returned, this could be modified later

"""


class Gridworld: # Environment
    def __init__(self, 
        D_in, 
        n_obj, 
        min_num, 
        max_num,
        game_over_reward, 
        step_reward, 
        current_goal_reward, 
        final_goal_reward,
        int_goal_reward, 
        int_step_reward, 
        int_wrong_goal_reward):

        # creates a square gridworld
        self.D_in = D_in
        self.n_obj = n_obj
        self.min_num = min_num
        self.max_num = max_num
        self.grid_mat = torch.zeros((self.D_in, self.D_in),dtype=torch.float32)

        # objects, gridworld and goals
        self.original_objects = self.create_objects()
        self.object_idxs = self.place_objects()
        self.agent_loc = torch.zeros((1,2),dtype=torch.int)
        self.agent_loc[0,:] = self.random_start()
        self.grid_mat_original = copy.deepcopy(self.grid_mat)
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = self.current_objects[0]

        # actions
        self.allowable_actions = {}
        self.allowable_action_idxs = {}
        self.all_actions = ['U','D','R','L']
        self.set_actions()

        # rewards
        self.game_over_reward = game_over_reward
        self.step_reward = step_reward
        self.current_goal_reward = current_goal_reward
        self.final_goal_reward = final_goal_reward
        self.int_goal_reward = int_goal_reward
        self.int_step_reward = int_step_reward
        self.int_wrong_goal_reward = int_wrong_goal_reward

    def set_actions(self):
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        for i in range(self.D_in):
            for j in range(self.D_in):
                if i != 0 and i != self.D_in-1 and j != 0 and j != self.D_in-1:
                    self.allowable_actions[(i,j)] = self.all_actions
                    self.allowable_action_idxs[(i,j)] = [0,1,2,3]
                else:
                    if i == 0 and j != 0 and j != self.D_in-1:
                        self.allowable_actions[(i,j)] = ['D','R','L']
                        self.allowable_action_idxs[(i,j)] = [1,2,3]
                    if i == self.D_in-1 and j != 0 and j != self.D_in-1:
                        self.allowable_actions[(i,j)] = ['U','R','L']
                        self.allowable_action_idxs[(i,j)] = [0,2,3]
                    if j == 0 and i != 0 and i != self.D_in-1:
                        self.allowable_actions[(i,j)] = ['U','D','R']
                        self.allowable_action_idxs[(i,j)] = [0,1,2]
                    if j == self.D_in-1 and i != 0 and i != self.D_in-1:
                        self.allowable_actions[(i,j)] = ['U','D','L']
                        self.allowable_action_idxs[(i,j)] = [0,1,3]
                    if i == 0 and j == 0:
                        self.allowable_actions[(i,j)] = ['D','R']
                        self.allowable_action_idxs[(i,j)] = [1,2]
                    if i == self.D_in-1 and j == 0:
                        self.allowable_actions[(i,j)] = ['U','R']
                        self.allowable_action_idxs[(i,j)] = [0,2]
                    if j == self.D_in-1 and i == 0:
                        self.allowable_actions[(i,j)] = ['D','L']
                        self.allowable_action_idxs[(i,j)] = [1,3]
                    if j == self.D_in-1 and i == self.D_in-1:
                        self.allowable_actions[(i,j)] = ['U','L']
                        self.allowable_action_idxs[(i,j)] = [0,3]

    def update_target_goal(self):
        if self.current_objects:
            self.current_target_goal = self.current_objects[0]

    def create_objects(self):
        arr = np.arange(self.min_num, self.max_num+1)
        objects = np.random.choice(arr, size=self.n_obj, replace=False, p=None)
        objects = [int(element) for element in objects]   
        objects = list(objects)
        objects.sort(reverse=False)
        return objects

    def place_objects(self):
        '''
        -places the objects
        -store the object's indexes
        -places the agent
        '''
        small_matrix_dim = (self.D_in+1)/2
        a = np.arange(0,small_matrix_dim**2, dtype=int)
        idx_1d = np.random.choice(a, size=self.n_obj, replace=False, p=None)
        idx_2d = [[2*math.floor(idx/small_matrix_dim), 2*int(idx%small_matrix_dim)] for idx in idx_1d]
        object_idxs = []
        for counter, idx in enumerate(idx_2d):
            self.grid_mat[idx[0], idx[1]] = self.original_objects[counter]
            object_idxs.append((idx[0],idx[1]))
        return object_idxs

    def reset(self): 
        # the function creates the same gridworld as the original
        self.grid_mat = copy.deepcopy(self.grid_mat_original)
        # self.grid_flat = np.ravel(self.grid_mat).reshape((1,self.D_in*self.D_in)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = self.current_objects[0]
        self.agent_loc[0,:] = self.random_start()
        return self.agent_loc, self.grid_mat

    def reset_total(self): # whole gridworld is created from the beginning and agent is 
                    # placed in a random position
        self.original_objects = self.create_objects()
        self.grid_mat = torch.zeros((self.D_in, self.D_in),dtype=torch.float32)
        self.object_idxs = self.place_objects()
        # self.grid_flat = np.ravel(self.grid_mat).reshape((1,self.D_in*self.D_in)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = self.current_objects[0]
        self.agent_loc[0,:] = self.random_start()
        return self.agent_loc, self.grid_mat

    def random_start(self):
        start = torch.zeros([1,2], dtype=torch.int)

        done = False
        while not done:
            i = np.random.choice(self.D_in)
            j = np.random.choice(self.D_in)
            if self.grid_mat[i,j] == 0:
                done = True
                start[0,0] = i
                start[0,1] = j
        return start


    def remove_object(self, i, j):
        self.grid_mat[i,j] = 0
        return self.current_objects.pop(0)


    def set_state(self, s):
        self.agent_loc[0,:] = s[0,:]
    
    def print_grid(self):
        print (self.grid_mat)
    
    def current_state(self):
        return self.agent_loc

    def step(self, action_idx):
        i = self.agent_loc[0,0].item()
        j = self.agent_loc[0,1].item()
        action = self.all_actions[action_idx]
    # check if legal move first, if not, nothing happens!
        if action in self.allowable_actions[(i,j)]:
            if   action == 'U':
                self.agent_loc[0,0] += -1
            elif action == 'L':
                self.agent_loc[0,1] += -1
            elif action == 'D':
                self.agent_loc[0,0] += 1
            elif action == 'R':
                self.agent_loc[0,1] += 1

            self.extr_reward(self.agent_loc)


        return self.agent_loc, self.grid_mat

    def int_reward(self, agent_state, meta_goal):
        i = agent_state[0,0]
        j = agent_state[0,1]
        element = self.grid_mat[i,j].item()

        if element == 0:
            reward = self.int_step_reward
        else:
            if element == meta_goal:
                reward = self.int_goal_reward
            else:
                reward = self.int_wrong_goal_reward
        return reward

    def extr_reward(self, agent_state):
        i = agent_state[0,0]
        j = agent_state[0,1]
        element = self.grid_mat[i,j].item()

        num_goals_left = len(self.current_objects)
        final_goal = False
        if num_goals_left == 1:
            final_goal = True
        
        if element == 0:
            reward = self.step_reward
        else:
            if element == self.current_target_goal:
                reward = self.current_goal_reward
            else:
                reward = self.game_over_reward
        # else: # if the last goal is picked
        #     if element == self.original_objects[-1]:
        #         reward = self.final_goal_reward
        #     else: 
        #         reward = self.game_over_reward
        return reward

    def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.agent_loc[0,0] += 1
        elif action == 'D':
            self.agent_loc[0,0] -= 1
        elif action == 'R':
            self.agent_loc[0,1] -= 1
        elif action == 'L':
            self.agent_loc[0,1] += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())

    def is_terminal(self, agent_state):
        # it is terminal either when it's game over or it has solved the game
        
        i = agent_state[0,0]
        j = agent_state[0,1]
        num_goals_left = len(self.current_objects)
        element = self.grid_mat[i,j].item()
        game_won = False
        game_over = False

        final_stage = False
        if num_goals_left == 1:
            # it will only get here when all the previous goals 
            # have been picked in the right order
            final_stage = True        
        
        if element == 0:
            pass
        elif final_stage:
            if element == self.current_target_goal:
                game_won = True
            else:
                pass
        else: 
            if element == self.current_target_goal:
                pass
            else: 
                game_over = True

        return game_over, game_won

    def game_over(self, state):
        pass

    def all_states(self):
        return set(self.allowable_actions.keys() + self.rewards.keys()) 