import numpy as np
import math
import copy
from pdb import set_trace


""" NOTES
- in take_actions, right now if the action is illegal, nothing happens and no reward is returned, this could be modified later

"""


class Gridworld: # Environment
    def __init__(self, n_dim, start, n_obj, min_num, max_num,
        not_moving_reward, terminal_reward, step_reward, goal_reward, final_goal_reward,
        intrinsic_goal_reward, intrinsic_step_reward, intrinsic_wrong_goal_reward):
        # creates a square gridworld
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.min_num = min_num
        self.max_num = max_num
        self.start = start
        self.i = start[0]
        self.j = start[1]
        self.grid_mat = np.zeros((self.n_dim, self.n_dim),dtype=int)

        # actions objects and goals
        self.original_objects = self.create_objects() # it's a list
        self.place_objects()
        self.grid_flattened = np.ravel(self.grid_mat).reshape((1,n_dim*n_dim)) # 1D array
        self.current_objects = copy.deepcopy(self.original_objects)
        self.allowable_actions = {}
        self.set_actions()
        self.selected_goals = []
        self.target_current_goal = self.original_objects[len(self.selected_goals)]
        self.all_actions = ['U','D','R','L']

        # rewards
        self.not_moving_reward = not_moving_reward
        self.terminal_reward = terminal_reward
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.final_goal_reward = final_goal_reward
        self.intrinsic_goal_reward = intrinsic_goal_reward
        self.intrinsic_step_reward = intrinsic_step_reward
        self.intrinsic_wrong_goal_reward = intrinsic_wrong_goal_reward

    def set_actions(self):
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                if i != 0 and i != self.n_dim-1 and j != 0 and j != self.n_dim-1:
                    self.allowable_actions[(i,j)] = ['U','D','R','L']
                else:
                    if i == 0 and j != 0 and j != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['D','R','L']
                    if i == self.n_dim-1 and j != 0 and j != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','R','L']
                    if j == 0 and i != 0 and i != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','D','R']
                    if j == self.n_dim-1 and i != 0 and i != self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','D','L']
                    if i == 0 and j == 0:
                        self.allowable_actions[(i,j)] = ['D','R']
                    if i == self.n_dim-1 and j == 0:
                        self.allowable_actions[(i,j)] = ['U','R']
                    if j == self.n_dim-1 and i == 0:
                        self.allowable_actions[(i,j)] = ['D','L']
                    if j == self.n_dim-1 and i == self.n_dim-1:
                        self.allowable_actions[(i,j)] = ['U','L']

    def update_target_goal(self):
        self.target_current_goal = self.original_objects[len(self.selected_goals)]

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

    def reset(self):
        return self.start

    def intrinsic_critique(self,state, goal):
        i = state[0]
        j = state[1]
        element = self.grid_mat[i,j]
        goal_reached = False

        if element == 0:
            reward = self.intrinsic_step_reward
        elif element == goal:
            reward = self.intrinsic_goal_reward
            goal_reward = True
        else:
            reward = self.intrinsic_wrong_goal_reward
            
        return reward, goal_reached

    def extrinsic_reward(self,i,j):
        num_goals_selected = len(self.selected_goals)
        final_goal = False
        if num_goals_selected == self.n_obj:
            final_goal = True        

        element = self.grid_mat[i,j]
        if element == 0:
            reward = -self.step_reward
        elif not final_goal:
            if element == self.target_current_goal:
                reward = self.goal_reward
            else:
                reward = self.terminal_reward
        else: # if the last goal is picked
            if element == self.original_objects[-1]:
                reward = self.final_goal_reward
            else: 
                reward = self.terminal_reward
        return reward

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
    
    def print_grid(self):
        print (self.grid_mat)
    
    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        num_goals_selected = len(self.selected_goals)
        final_goal = False
        if num_goals_selected == self.n_obj:
            final_goal = True        

        i = s[0]
        j = s[1]
        element = self.grid_mat[i,j]
        if element == 0:
            return False
        elif not final_goal:
            if element == self.target_current_goal:
                return False
            else:
                return True
        else: # if the last goal is picked
            if element == self.original_objects[-1]:
                return False
            else: 
                return True 
        
    
    def take_action(self, action_idx):
        action = self.all_actions[action_idx]
    # check if legal move first, if not, nothing happens!
        if action in self.allowable_actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1

            return self.extrinsic_reward(self.i, self.j), np.asarray([self.i, self.j])
        else:
            return self.not_moving_reward, np.asarray([self.i, self.j])

    def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.allowable_actions

    def all_states(self):
        return set(self.allowable_actions.keys() + self.rewards.keys()) 