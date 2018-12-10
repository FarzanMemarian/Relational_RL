# used part of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import random
import numpy as np
import copy
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import SGD, RMSprop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('../')
from envs.gridworld1 import Gridworld
from utils import utils
from pdb import set_trace

# DEFAULT ARCHITECTURE FOR THE META cntr
default_meta_batch_size = 1000
default_meta_epsilon = 1.0
default_meta_memory_size = 10000

# DEFAULT ARCHITECTURES FOR THE LOWER LEVEL cntr/cntr
default_batch_size = 1000
default_gamma = 0.975
default_epsilon = 1.0
default_tau = 0.001
default_cntr_memory_size = 10000


class cntr_class(nn.Module):

    def __init__(self, final_conv_dim=3):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 6, kernel_size=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=2)
        # an affine operation: y = Wx + b
        self.final_conv_dim = final_conv_dim
        self.fc1 = nn.Linear(16 * final_conv_dim * final_conv_dim, 100)
        self.fc2 = nn.Linear(100, 40) 
        self.fc3 = nn.Linear(40, 4)   

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    


class meta_class(nn.Module):

    def __init__(self, final_conv_dim=3):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 6, kernel_size=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=2)
        # an affine operation: y = Wx + b
        self.final_conv_dim = final_conv_dim
        self.fc1 = nn.Linear(16 * final_conv_dim * final_conv_dim, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, exp):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class hDQN:

    def __init__(self, 
                env, 
                batch_size=default_batch_size,
                meta_batch_size=default_meta_batch_size, 
                gamma=default_gamma,
                meta_epsilon=default_meta_epsilon, 
                epsilon=default_epsilon, 
                tau = default_tau,
                cntr_memory_size = default_cntr_memory_size,
                meta_memory_size = default_meta_memory_size):

        self.env = env
        self.goal_selected = np.zeros(len(self.env.original_objects))
        self.goal_success = np.zeros(len(self.env.original_objects))
        self.meta_epsilon = meta_epsilon
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.gamma = gamma
        self.target_tau = tau
        
        self.cntr_memory_size = cntr_memory_size
        self.meta_memory_size = meta_memory_size        
        self.cntr_memory = ReplayMemory(self.cntr_memory_size)
        self.meta_memory = ReplayMemory(self.meta_memory_size)

        
        self.policy_meta_net = meta_class()
        self.target_meta_net = meta_class()
        self.target_meta_net.load_state_dict(self.policy_meta_net.state_dict())
        self.target_meta_net.eval()
        self.meta_optimizer = optim.SGD(self.policy_meta_net.parameters(), lr=0.001)
        self.meta_criterion = nn.MSELoss()
        
        self.policy_cntr_net = cntr_class()
        self.target_cntr_net = cntr_class()
        self.target_cntr_net.load_state_dict(self.policy_cntr_net.state_dict())
        self.target_cntr_net.eval()
        self.cntr_optimizer = optim.SGD(self.policy_cntr_net.parameters(), lr=0.001)
        self.cntr_criterion = nn.MSELoss()

    def select_goal(self, agent_state):
        if self.meta_epsilon < random.random():
            with torch.no_grad():
                Q = []
                for goal in self.env.current_objects:
                    meta_input = utils.meta_input(self.env.D_in, agent_state, self.env.grid_mat, goal)
                    pred, _ = self.Q_meta(meta_input, False)
                    Q.append(pred.item())
                goal_idx = np.argmax(Q)
                goal = self.env.current_objects[goal_idx]

        else:
            print("Exploring ...............")
            goal = self.random_goal_selection()
        # update environment
        self.env.selected_goals.append(goal)
        return goal

    def random_goal_selection(self):
        # Don't call this function directly, it would always be called from select_goal()
        goal_idx = int(np.random.choice(len(self.env.current_objects)))
        goal = self.env.current_objects[goal_idx]
        return goal


    def select_action(self, agent_state, goal):
        i = agent_state[0,0].item()
        j = agent_state[0,1].item()
        if random.random() > self.epsilon:
            # print("cntr selected action")
            # ensures that only actions that cause movement are chosen
            with torch.no_grad():
                cntr_input = utils.cntr_input(self.env.D_in, agent_state, self.env.grid_mat, goal)
                action_probs, _ = self.Q_cntr(cntr_input, False) 
                allowable_action_idxs = self.env.allowable_action_idxs[i, j]
                allowable_action_probs = action_probs[0,allowable_action_idxs]
                allowable_action_probs_max_idx = np.argmax(allowable_action_probs.numpy())
                action_idx = allowable_action_idxs[allowable_action_probs_max_idx]
                action = self.env.all_actions[action_idx]
                # print ("\n \naction_probs: {}".format(action_probs))
                # print ("allowable_action_idxs: {}".format(allowable_action_idxs))
                # print ("allowable_action_probs: {}".format(allowable_action_probs))
                # print ("allowable_action_probs_max_idx: {}".format(allowable_action_probs_max_idx))
                # print ("action_idx: {}".format(action_idx))
            # set_trace()
        else:
            action_idx, action = self.random_action_selection()
        return action_idx, action

    def random_action_selection(self):
        # print("random action selected")
        i = self.env.agent_loc[0,0].item() 
        j = self.env.agent_loc[0,1].item()
        allowable_action_idxs = self.env.allowable_action_idxs[i,j]
        action_idx = int(np.random.choice(allowable_action_idxs))
        action = self.env.all_actions[action_idx]
        return action_idx, action

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.push(experience)

        else:
            self.cntr_memory.push(experience)


    def Q_cntr(self, cntr_input, target):
        if target:
            try:
                Q_policys = self.target_cntr_net(cntr_input) 
            except Exception as e:
                cntr_input = torch.unsqueeze(cntr_input, dim=0)
                Q_policys = self.target_cntr_net(cntr_input) 
            return Q_policys, cntr_input
        else:
            try:
                Q_policys = self.policy_cntr_net(cntr_input) 
            except Exception as e:
                cntr_input = torch.unsqueeze(cntr_input, dim=0)
                Q_policys = self.policy_cntr_net(cntr_input)
            return Q_policys, cntr_input

    def Q_meta(self, meta_input, target):
        if target:
            try:
                Q_policys = self.target_meta_net(meta_input)
            except Exception as e:
                meta_input = torch.unsqueeze(meta_input, dim=0)
                Q_policys = self.target_meta_net(meta_input)
            return Q_policys, meta_input
        else:
            try:
                Q_policys = self.policy_meta_net(meta_input)
            except Exception as e:
                meta_input = torch.unsqueeze(meta_input, dim=0)
                Q_policys = self.policy_meta_net(meta_input)
            return Q_policys, meta_input



    def _update_cntr(self):
        if len(self.cntr_memory) < self.batch_size:
            return

        sample_size = min(self.batch_size, len(self.cntr_memory))
        exps = self.cntr_memory.sample(sample_size)
        
        state_tensors = torch.cat([torch.unsqueeze(exp.agent_env_goal_cntr, 0) for exp in exps])
        non_terminal_mask = torch.tensor(tuple(map(lambda s: s != True, [exp.cntr_done for exp in exps])))
        try:
            next_state_non_terminals = torch.cat([torch.unsqueeze(exp.next_agent_env_goal_cntr, 0) 
            for exp in exps if exp.cntr_done != True])
        except:
            print("all states done ......")
        next_state_Vs = torch.zeros(sample_size)
        next_state_Vs[non_terminal_mask] = self.Q_cntr(next_state_non_terminals, 
                                           target=True)[0].max(1)[0].detach()
        
        # action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor([exp.int_reward for exp in exps], dtype=torch.float32)
        action_batch = torch.unsqueeze(torch.tensor([exp.action_idx for exp in exps]), 1)

        Q_policys = self.Q_cntr(state_tensors, target=False)[0].gather(1, action_batch)

        
        try:
            Q_targets = (next_state_Vs * self.gamma) + reward_batch
        except:
            print("Q targets problems...")
            set_trace()

        Q_targets = torch.unsqueeze(Q_targets, 1)

        self.cntr_optimizer.zero_grad()
        loss = self.cntr_criterion(Q_policys, Q_targets)
        loss.backward()
        self.cntr_optimizer.step()
        #Update target network
        with torch.no_grad():
            cntr_weights = self.policy_cntr_net.parameters()
            cntr_target_weights = self.target_cntr_net.parameters()

            for layer_w, target_layer_w in zip(cntr_weights, cntr_target_weights):
                target_layer_w = self.target_tau * layer_w + (1 - self.target_tau) * target_layer_w

    def _update_meta(self):
        # ["agent_env_state_0", "goal", "reward", "next_agent_env_state", "next_available_goals", "done"]

        if len(self.meta_memory) <= 5:
            return

        sample_size = min(self.meta_batch_size, len(self.meta_memory))
        exps = self.meta_memory.sample(sample_size)
        # exps = self.meta_memory[-sample_size:]
        D_in = self.env.D_in
        state_tensors = torch.cat([torch.unsqueeze(utils.meta_input2(D_in, exp.agent_env_state_0, 
                            exp.goal), 0) for exp in exps])
        Q_policys = self.Q_meta(state_tensors, target=False)[0]


        reward_batch = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

        # non_terminal_mask = torch.tensor(tuple(map(lambda s: s != True, [exp.terminal for exp in exps])))
        # try:
        #     next_state_non_terminals = torch.cat([torch.unsqueeze(utils.meta_input2(D_in, 
        #         exp.next_agent_env_state, exp.goal), 0) for exp in exps if exp.terminal != True])
        #     next_state_Vs = self.Q_meta(next_state_non_terminals, target=True)[0].detach()
        # except:
        #     pass


        Q_targets = torch.zeros((sample_size, 1))

        next_state_Vs = torch.zeros(sample_size)
        # next_state_Vs[non_terminal_mask] = self.Q_cntr(next_state_non_terminals, 
        #     target=True)[0].max(1)[0].detach()
        # Q_targets = reward_batcht + (next_state_Vs * self.gamma) 
        
        Q_targets[:,0] = reward_batch

        for i, exp in enumerate(exps):
            if not exp.terminal:
                # this block finds the max Q in the next state for this particular experiment
                # if exp.terminal is true, it means that the next state is terminal and we have 
                # Q(s,.)=0 if s is terminal 
                
                try:
                    intermediate_tensor = torch.cat([torch.unsqueeze(utils.meta_input2(D_in, 
                                    exp.next_agent_env_state, next_goal), 0)
                                    for next_goal in exp.next_available_goals])
                except:
                    print("PROBLEM")
                    print("PROBLEM")
                    print("PROBLEM")
                    print("PROBLEM")
                    print("PROBLEM")
                    set_trace()
                    print("stop here")

                next_state_V = max(self.Q_meta(intermediate_tensor, target=True)[0]).item()
                Q_targets[i,0] +=  self.gamma * (next_state_V)
                # Q_targets = (next_state_Vs * self.gamma) + reward_batch


        self.meta_optimizer.zero_grad()
        loss = self.meta_criterion(Q_policys, Q_targets)
        loss.backward()
        self.meta_optimizer.step()
        #Update target network
        with torch.no_grad():
            meta_weights = self.policy_meta_net.parameters()
            meta_target_weights = self.target_meta_net.parameters()

            for layer_w, target_layer_w in zip(meta_weights, meta_target_weights):
                target_layer_w = self.target_tau * layer_w + (1 - self.target_tau) * target_layer_w


    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update_cntr()