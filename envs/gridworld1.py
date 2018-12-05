import numpy as np
import math
import copy
from pdb import set_trace


""" NOTES
- in take_actions, right now if the action is illegal, nothing happens and no reward is returned, this could be modified later

"""


class Gridworld: # Environment
    def __init__(self, 
        n_dim, 
        start, 
        n_obj, 
        min_num, 
        max_num,
        not_moving_reward, 
        game_over_reward, 
        step_reward, 
        current_goal_reward, 
        final_goal_reward,
        intrinsic_goal_reward, 
        intrinsic_step_reward, 
        intrinsic_wrong_goal_reward):

        # creates a square gridworld
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.min_num = min_num
        self.max_num = max_num
        self.start = start
        self.grid_mat = np.zeros((self.n_dim, self.n_dim),dtype=int)
        self.state = np.zeros((1,2),dtype=int)

        # objects, gridworld and goals
        self.state[0,:] = self.start[0,:]
        self.original_objects = self.create_objects() # it's a list
        self.place_objects()
        self.grid_mat_original = copy.deepcopy(self.grid_mat)
        self.grid_flat = np.ravel(self.grid_mat).reshape((1,n_dim*n_dim)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = None

        # actions
        self.allowable_actions = {}
        self.allowable_action_idxs = {}
        self.all_actions = ['U','D','R','L']
        self.set_actions()

        # rewards
        self.not_moving_reward = not_moving_reward
        self.game_over_reward = game_over_reward
        self.step_reward = step_reward
        self.current_goal_reward = current_goal_reward
        self.final_goal_reward = final_goal_reward
        self.intrinsic_goal_reward = intrinsic_goal_reward
        self.intrinsic_step_reward = intrinsic_step_reward
        self.intrinsic_wrong_goal_reward = intrinsic_wrong_goal_reward

    def set_actions(self):
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
        self.current_target_goal = self.current_objects[0]

    def create_objects(self):
        a = np.arange(self.min_num, self.max_num+1)
        objects = np.random.choice(a, size=self.n_obj, replace=False, p=None)
        objects = list(objects)
        objects.sort(reverse=False)
        return objects

    def place_objects(self):
        small_matrix_dim = (self.n_dim+1)/2
        a = np.arange(0,small_matrix_dim**2, dtype=int)
        idx_1d = np.random.choice(a, size=self.n_obj, replace=False, p=None)
        idx_2d = [[2*math.floor(idx/small_matrix_dim), 2*int(idx%small_matrix_dim)] for idx in idx_1d]
        for counter, idx in enumerate(idx_2d):
            self.grid_mat[idx[0], idx[1]] = self.original_objects[counter]

    def reset(self): # same grid-world is recreated
        self.state[0,:] = self.start[0,:]
        self.grid_mat = copy.deepcopy(self.grid_mat_original)
        self.grid_flat = np.ravel(self.grid_mat).reshape((1,self.n_dim*self.n_dim)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = None
        return self.start, self.grid_flat

    def reset2(self): # whole gridworld is created from the beginning. 
        self.state[0,:] = self.start[0,:]
        self.original_objects = self.create_objects() # it's a list
        self.place_objects()
        self.grid_mat_original = copy.deepcopy(self.grid_mat)
        self.grid_flat = np.ravel(self.grid_mat).reshape((1,self.n_dim*self.n_dim)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.selected_goals = []
        self.current_target_goal = None
        return self.start

    def intrinsic_critique(self,state, goal):
        i = state[0,0]
        j = state[0,1]
        element = self.grid_mat[i,j]
        goal_reached = False

        if element == 0:
            reward = self.intrinsic_step_reward
        elif element == goal:
            reward = self.intrinsic_goal_reward
            goal_reached = True
        else:
            reward = self.intrinsic_wrong_goal_reward
            
        return reward, goal_reached

    def extrinsic_reward(self):
        i = self.state[0,0]
        j = self.state[0,1]
        num_goals_left = len(self.current_objects)
        final_goal = False
        if num_goals_left == 1:
            final_goal = True    

        element = self.grid_mat[i,j]
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

                

    def remove_occupied_object(self):
        self.grid_mat[self.state[0,0], self.state[0,1]] = 0
        self.current_objects.pop(0)
        self.grid_flat = np.ravel(self.grid_mat).reshape((1,self.n_dim*self.n_dim)) # 1D array


    def set_state(self, s):
        self.state[0,:] = s[0,:]
    
    def print_grid(self):
        print (self.grid_mat)
    
    def current_state(self):
        return self.state

    def take_action(self, action_idx):
        action = self.all_actions[action_idx]
    # check if legal move first, if not, nothing happens!
        if action in self.allowable_actions[(self.state[0,0], self.state[0,1])]:
            if   action == 'U':
                self.state[0,0] += -1
            elif action == 'L':
                self.state[0,1] += -1
            elif action == 'D':
                self.state[0,0] += 1
            elif action == 'R':
                self.state[0,1] += 1

            reward = self.extrinsic_reward()
        else:
            # this part should never be executed if the actions are only picked amont the allowable actions
            reward = self.not_moving_reward

        return reward, self.state, self.grid_flat

    def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.state[0,0] += 1
        elif action == 'D':
            self.state[0,0] -= 1
        elif action == 'R':
            self.state[0,1] -= 1
        elif action == 'L':
            self.state[0,1] += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())

    def is_terminal(self, state):
        # it is terminal either when it's game over or it has solved the game
        
        i = state[0,0]
        j = state[0,1]
        num_goals_left = len(self.current_objects)
        element = self.grid_mat[i,j]
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