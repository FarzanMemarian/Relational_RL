# used part of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import random
import numpy as np
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
default_memory_size = 10000


class cntr_class(nn.Module):

    def __init__(self, final_conv_dim=3):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 6, 2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        # an affine operation: y = Wx + b
        self.final_conv_dim = final_conv_dim
        self.fc1 = nn.Linear(16 * final_conv_dim * final_conv_dim, 100)
        self.fc2 = nn.Linear(100, 40) 
        self.fc3 = nn.Linear(40, 4)   

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
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
        self.conv1 = nn.Conv2d(2, 6, 2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        # an affine operation: y = Wx + b
        self.final_conv_dim = final_conv_dim
        self.fc1 = nn.Linear(16 * final_conv_dim * final_conv_dim, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
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


class hDQN:

    def __init__(self, 
                env, 
                batch_size=default_batch_size,
                meta_batch_size=default_meta_batch_size, 
                gamma=default_gamma,
                meta_epsilon=default_meta_epsilon, 
                epsilon=default_epsilon, 
                tau = default_tau,
                memory_size = default_memory_size,
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
        self.memory = []
        self.meta_memory = []
        self.memory_size = memory_size
        self.meta_memory_size = meta_memory_size
        self.meta_net = meta_class()
        self.target_meta_net = meta_class()
        self.cntr_net = cntr_class()
        self.target_cntr_net = cntr_class()



    def select_goal(self, state):
        if self.meta_epsilon < random.random():
            Q = []
            meta_input = torch.tensor([2, self.input_dim, self.input_dim], dtype=torch.int)
            for goal in self.env.current_objects:
                meta_input[0] = goal
                meta_input[1] = state
                pred = self.meta_net(meta_input)
                Q.append(pred)
            goal_idx = np.argmax(Q)
            goal = self.env.current_objects[goal_idx]
        else:
            print("Exploring")
            goal = self.random_goal_selection()

        # update environment
        self.env.selected_goals.append(goal)
        self.env.update_target_goal()
        return goal

    def random_goal_selection(self):
        # Don't call this function directly, it would always be called from select_goal()
        goal_idx = np.random.choice(len(self.env.current_objects))
        goal = self.env.original_objects[goal_idx]
        return goal


    def select_action(self, state):
        state = torch.from_numpy(state)
        if random.random() > self.epsilon:
            # print("cntr selected action")
            # ensures that only actions that cause movement are chosen
            action_probs = self.cntr_net(state)        
            allowable_action_idxs = self.env.allowable_action_idxs[self.env.state[0,0], self.env.state[0,1]]
            allowable_action_probs = action_probs[0,allowable_action_idxs]
            allowable_action_probs_max_idx = np.argmax(allowable_action_probs)
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
        allowable_action_idxs = self.env.allowable_action_idxs[self.env.state[0,0], self.env.state[0,1]]
        action_idx = np.random.choice(allowable_action_idxs)
        action = self.env.all_actions[action_idx]
        return action_idx, action

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
            if len(self.meta_memory) > self.meta_memory_size:
                self.meta_memory = self.meta_memory[-self.meta_memory_size:]
        else:
            self.memory.append(experience)
            if len(self.memory) > self.memory_size:
                self.memory = self.memory[-self.memory_size:]

    def Q_cntr(self, state, target):
        if target:
            try:
                Q_preds = self.target_cntr_net(state)
            except Exception as e:
                state_tensors = np.expand_dims(state_tensors, axis=0)
                Q_preds = self.target_cntr_net(state)
            return Q_preds, state_tensors
        else:
            try:
                Q_preds = self.cntr_net(state)
            except Exception as e:
                state_tensors = np.expand_dims(state_tensors, axis=0)
                Q_preds = self.cntr_net(state)
            return Q_preds, state

    def Q_meta(self, state, target):
        if target:
            try:
                Q_preds = self.target_meta_net(state)
            except Exception as e:
                state = np.expand_dims(state, axis=0)
                Q_preds = self.target_meta_net(state)
            return Q_preds, state
        else:
            try:
                Q_preds = self.meta_net(state)
            except Exception as e:
                state = np.expand_dims(state, axis=0)
                Q_preds = self.meta_net(state)
            return Q_preds, state



    def _update_cntr(self):
        sample_size = min(self.batch_size, len(self.memory))
        exps = [random.choice(self.memory) for _ in range(sample_size)]

        state_tensors = torch.Tensor((sample_size, 2, self.input_dim, self.input_dim))
        torch.cat([torch.unsqueeze(exp.state, 0) for exp in exps], out=state_tensors)
        next_state_tensors = torch.Tensor((sample_size, 2, self.input_dim, self.input_dim))
        torch.cat([torch.unsqueeze(exp.next_state, 0) for exp in exps], out=state_tensors)

        Q_preds, state_tensors  = self.Q_cntr(state_tensors, target=False)
        next_state_Q_preds, next_state_tensors = self.Q_cntr(next_state_tensors, target=True)
        Q_targets = Q_preds
        for i, exp in enumerate(exps):
            Q_targets[i,exp.action] = exp.reward
            if not exp.cntr_done:
                # if exp.cntr_done is true, it means that the next state is terminal and we have 
                # Q(s,.)=0 if s is terminal
                Q_targets[i,exp.action] += self.gamma * max(next_state_Q_preds[i])
        

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.cntr_net.parameters(), lr=0.001, momentum=0.9)        
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(Q_preds, Q_targets)
        loss.backward()
        optimizer.step()

        #Update target network
        cntr_weights = self.cntr_net.parameters()
        cntr_target_weights = self.target_cntr_net.parameters()
        for i in range(len(cntr_weights)):
            cntr_target_weights[i] = self.target_tau * cntr_weights[i] + \
                                           (1 - self.target_tau) * cntr_target_weights[i]
        self.target_cntr_net.set_weights(cntr_target_weights)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            sample_size = min(self.meta_batch_size, len(self.meta_memory))
            exps = [random.choice(self.meta_memory) for _ in range(sample_size)]
            state_tensors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], 
                                    axis=1) for exp in exps]))

            Q_preds, state_tensors = self.Q_meta(state_tensors, target=False)

            Q_targets = np.zeros((sample_size,1))
            for i, exp in enumerate(exps):
                Q_targets[i,0] = exp.reward

                if not exp.done:
                    # this block finds the max Q in the next state for this particular experiment
                    # if exp.done is true, it means that the next state is terminal and we have 
                    # Q(s,.)=0 if s is terminal 
                    next_state_tensors = np.squeeze(np.asarray([np.concatenate([exp.state, next_goal.reshape((1,1))], 
                                    axis=1) for next_goal in exp.next_available_goals]))
                    next_state_Q_preds, next_state_tensors = self.Q_meta(next_state_tensors, target=True)
                    Q_targets[i,0] += self.gamma * max(next_state_Q_preds)

            self.meta_cntr.fit(state_tensors, Q_targets)
            
            #Update target network
            meta_weights = self.meta_cntr.get_weights()
            meta_target_weights = self.target_meta_cntr.get_weights()
            for i in range(len(meta_weights)):
                meta_target_weights[i] = self.target_tau * meta_weights[i] + (1 - self.target_tau) * meta_target_weights[i]
            self.target_meta_cntr.set_weights(meta_target_weights)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update_cntr()