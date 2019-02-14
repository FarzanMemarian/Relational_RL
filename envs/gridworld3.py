import numpy as np
import math
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pdb import set_trace


""" NOTES
- in steps, right now if the action is illegal, nothing happens and no reward is returned, this could be modified later

"""


class Gridworld: # Environment
    def __init__(self, 
        n_dim, 
        n_obj, 
        min_num, 
        max_num,
        num_gridworlds,
        game_over_reward, 
        step_reward, 
        current_goal_reward, 
        final_goal_reward,
        int_goal_reward, 
        int_step_reward, 
        int_wrong_goal_reward,
        reset_type):

        # creates a square gridworld
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.min_num = min_num
        self.max_num = max_num
        self.num_gridworlds = num_gridworlds

        # objects, gridworld and goals
        self.agent_loc = torch.zeros((1,2),dtype=torch.int)
        self.env_state_original = None
        self.original_objects = None
        self.original_object_idxs = None
        self.is_env_state_created = False
        
        self.env_state = None
        self.current_objects = None
        self.current_object_idxs = None
        self.selected_goals = None
        self.current_target_goal = None
        self.current_target_goal_idx = None

        self.gridworld_finite_store = None

        if reset_type == "reset_finite":
            self.gridworld_finite_store = self._create_multiple_gridworlds()
        self.reset(reset_type)



        # actions
        self.allowable_actions = {}
        self.allowable_action_idxs = {}
        self.all_actions = ['U','D','R','L']
        self._set_actions()

        # rewards
        self.game_over_reward = game_over_reward
        self.step_reward = step_reward
        self.current_goal_reward = current_goal_reward
        self.final_goal_reward = final_goal_reward
        self.int_goal_reward = int_goal_reward
        self.int_step_reward = int_step_reward
        self.int_wrong_goal_reward = int_wrong_goal_reward

    def _set_actions(self):
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i != 0 and i != self.n_dim-1 and j != 0 and j != self.n_dim-1:
                    self.allowable_actions[(i,j)] = self.all_actions
                    self.allowable_action_idxs[(i,j)] = [0,1,2,3]
                else:
                    if i == 0 and j != 0 and j != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['D','R','L']
                        self.allowable_action_idxs[(i,j)] = [1,2,3]
                    if i == self.n_dim-1 and j != 0 and j != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','R','L']
                        self.allowable_action_idxs[(i,j)] = [0,2,3]
                    if j == 0 and i != 0 and i != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','D','R']
                        self.allowable_action_idxs[(i,j)] = [0,1,2]
                    if j == self.n_dim-1 and i != 0 and i != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','D','L']
                        self.allowable_action_idxs[(i,j)] = [0,1,3]
                    if i == 0 and j == 0:
                        self.allowable_actions[(i,j)] = ['D','R']
                        self.allowable_action_idxs[(i,j)] = [1,2]
                    if i == self.n_dim-1 and j == 0:
                        self.allowable_actions[(i,j)] = ['U','R']
                        self.allowable_action_idxs[(i,j)] = [0,2]
                    if j == self.n_dim-1 and i == 0:
                        self.allowable_actions[(i,j)] = ['D','L']
                        self.allowable_action_idxs[(i,j)] = [1,3]
                    if j == self.n_dim-1 and i == self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','L']
                        self.allowable_action_idxs[(i,j)] = [0,3]

    def update_target_goal(self):
        if self.current_objects:
            self.current_target_goal = self.current_objects[0]
            self.current_target_goal_idx = self.current_object_idxs[0]

    def _create_objects(self):
        arr = np.arange(self.min_num, self.max_num+1)
        objects = np.random.choice(arr, size=self.n_obj, replace=False, p=None)
        objects = [int(element) for element in objects]   
        objects.sort(reverse=False)
        return objects

    def _place_objects(self):
        '''
        -places the objects
        -store the object's indexes
        '''
        small_matrix_dim = (self.n_dim+1)/2
        a = np.arange(0,small_matrix_dim**2, dtype=int)
        idx_1d = np.random.choice(a, size=self.n_obj, replace=False, p=None)
        idx_2d = [[2*math.floor(idx/small_matrix_dim), 2*int(idx%small_matrix_dim)] for idx in idx_1d]
        object_idxs = []
        for counter, idx in enumerate(idx_2d):
            self.env_state_original[idx[0], idx[1]] = self.original_objects[counter]
            object_idxs.append((idx[0],idx[1]))
        return object_idxs


    def _create_multiple_gridworlds(self):
        state_list = []
        for _ in range(self.num_gridworlds):
            self.env_state_original = torch.zeros((self.n_dim, self.n_dim),dtype=torch.float32)
            self.original_objects = self._create_objects()
            self.original_object_idxs = self._place_objects()
            state_list.append([copy.deepcopy(self.env_state_original), copy.deepcopy(self.original_objects), 
                copy.deepcopy(self.original_object_idxs)])
        return state_list


    def reset(self, reset_type): 
        if reset_type == "reset":
        # the function creates the same gridworld as the original
            if not self.is_env_state_created:
                self.env_state_original = torch.zeros((self.n_dim, self.n_dim),dtype=torch.float32)
                self.original_objects = self._create_objects()
                self.original_object_idxs = self._place_objects()
                self.is_env_state_created = True

        elif reset_type == "reset_total":
            # whole gridworld is created from the beginning and agent is 
                        # placed in a random position
            self.env_state_original = torch.zeros((self.n_dim, self.n_dim),dtype=torch.float32)
            self.original_objects = self._create_objects()
            self.original_object_idxs = self._place_objects()

        elif reset_type == "reset_finite":
            # sample from a set of gridworlds created in _create_multiple_gridworlds
            sample = random.sample(self.gridworld_finite_store,1)[0]
            self.env_state_original = copy.deepcopy(sample[0])
            self.original_objects = copy.deepcopy(sample[1])
            self.original_object_idxs = copy.deepcopy(sample[2])

        self.env_state = copy.deepcopy(self.env_state_original)
        self.current_objects = copy.deepcopy(self.original_objects)
        self.current_object_idxs = copy.deepcopy(self.original_object_idxs)
        self.selected_goals = []
        self.current_target_goal = self.current_objects[0]
        self.current_target_goal_idx = self.current_object_idxs[0]
        self.agent_loc[0,:] = self._random_start()
        return copy.deepcopy(self.agent_loc), copy.deepcopy(self.env_state)


    def _random_start(self):
        start = torch.zeros([1,2], dtype=torch.int)

        done = False
        while not done:
            i = np.random.choice(self.n_dim)
            j = np.random.choice(self.n_dim)
            if self.env_state[i,j] == 0:
                start[0,0] = i
                start[0,1] = j
                done = True
        return start

    def remove_object(self, i, j):
        self.env_state[i,j] = 0
        self.current_object_idxs.pop(0)
        removed_object = self.current_objects.pop(0)
        return removed_object

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
        return copy.deepcopy(self.agent_loc), copy.deepcopy(self.env_state)

    def get_to_object(self, object):
        i, j = [self.original_object_idxs[item] for  item,obj in enumerate(self.original_objects) if obj == object][0]
        self.agent_loc[0,0] = i
        self.agent_loc[0,1] = j
        return copy.deepcopy(self.agent_loc), copy.deepcopy(self.env_state)

    def int_reward(self, agent_state, meta_goal):
        i = agent_state[0,0]
        j = agent_state[0,1]
        element = self.env_state[i,j].item()
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
        element = self.env_state[i,j].item()

        # num_goals_left = len(self.current_objects)
        # final_goal = False
        # if num_goals_left == 1:
        #     final_goal = True
        
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


    def is_terminal(self, agent_state):
        # it is terminal either when it's game over or it has solved the game
        
        i = agent_state[0,0]
        j = agent_state[0,1]
        num_goals_left = len(self.current_objects)
        element = self.env_state[i,j].item()
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

    def set_state(self, s):
        self.agent_loc[0,:] = s[0,:]
    
    def print_grid(self):
        print (self.env_state)
    
    def current_agent_state(self):
        return self.agent_loc

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

    def game_over(self, state):
        pass

    def all_states(self):
        return set(self.allowable_actions.keys() + self.rewards.keys()) 