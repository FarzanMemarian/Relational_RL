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
from src import transformer
from utils import utils
from pdb import set_trace
import time

# DEFAULT ARCHITECTURE FOR THE META cntr
default_meta_batch_size = 1000
default_meta_policy_temp = 1.0
default_meta_memory_size = 10000

# DEFAULT ARCHITECTURES FOR THE LOWER LEVEL cntr/cntr
default_batch_size = 1000
default_gamma = 0.975
default_epsilon = 1.0
default_tau = 0.001
default_cntr_memory_size = 10000


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class cntr_net_conv_MLP(nn.Module):

    def __init__(self, last_conv_depth=32, ndim=5):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 2
        self.stride = 1
        self.ndim = ndim
        self.last_conv_depth = last_conv_depth
        self.conv1 = nn.Conv2d(1,  16, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, last_conv_depth, kernel_size=self.kernel_size, stride=self.stride)
        self.bn2 = nn.BatchNorm2d(last_conv_depth)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(ndim,self.kernel_size, self.stride),self.kernel_size, self.stride)
        self.fc1 = nn.Linear(self.last_conv_depth * self.final_conv_dim * self.final_conv_dim + 1, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 4)


    def forward(self, x, g):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.append_goal(x,g)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def append_goal(self, x, g):
        x = torch.cat([x, g], 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    


class cntr_net_MLP(nn.Module):

    def __init__(self, last_conv_depth=32, ndim=5):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.kernel_size = 2
        self.stride = 1
        self.ndim = ndim
        self.last_conv_depth = last_conv_depth
        self.conv1 = nn.Conv2d(1,  16, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, last_conv_depth, kernel_size=self.kernel_size, stride=self.stride)
        self.bn2 = nn.BatchNorm2d(last_conv_depth)
        # an affine operation: y = Wx + b
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.final_conv_dim = conv2d_size_out(conv2d_size_out(ndim,self.kernel_size, self.stride),self.kernel_size, self.stride)
        self.fc1 = nn.Linear(1 * ndim * ndim + 1, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 4)


    def forward(self, x, g):
        x = x.view(-1, self.num_flat_features(x))
        x = self.append_goal(x,g)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def append_goal(self, x, g):
        x = torch.cat([x, g], 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class hDQN:

    def __init__(self, 
                env, 
                batch_size,
                meta_batch_size, 
                gamma,
                meta_policy_temp, 
                cntr_policy_temp, 
                tau,
                cntr_Transition,                
                cntr_memory_size,
                meta_Transition,
                meta_memory_size,
                meta_loss,
                meta_optimizer,
                meta_lr,
                cntr_loss,
                cntr_optimizer,
                cntr_lr,
                meta_clamp,
                cntr_clamp,
                cntr_network
                ):

        self.env = env
        self.meta_policy_temp = meta_policy_temp
        self.cntr_policy_temp = cntr_policy_temp
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.gamma = gamma
        self.target_tau = tau
        
        self.cntr_Transition = cntr_Transition
        self.cntr_memory_size = cntr_memory_size
        self.meta_Transition = meta_Transition
        self.meta_memory_size = meta_memory_size        
        self.cntr_memory = ReplayMemory(self.cntr_memory_size, self.cntr_Transition)
        self.meta_memory = ReplayMemory(self.meta_memory_size, self.meta_Transition)

        
        self.policy_meta_net = transformer.meta_class(n_dim=env.n_dim).to(device)
        self.target_meta_net = transformer.meta_class(n_dim=env.n_dim).to(device)
        self.target_meta_net.load_state_dict(self.policy_meta_net.state_dict())
        self.target_meta_net.eval()
        self.meta_optimizer, self.meta_criterion = self.set_optim(self.policy_meta_net.parameters(), 
            meta_optimizer, meta_loss, meta_lr)
        self.meta_clamp = meta_clamp

        self.cntr_network_name = cntr_network
        if self.cntr_network_name == "conv_MLP":
            self.policy_cntr_net = cntr_net_conv_MLP(ndim=env.n_dim).to(device)
            self.target_cntr_net = cntr_net_conv_MLP(ndim=env.n_dim).to(device)
        if self.cntr_network_name == "MLP":
            self.policy_cntr_net = cntr_net_MLP(ndim=env.n_dim).to(device)
            self.target_cntr_net = cntr_net_MLP(ndim=env.n_dim).to(device)

        self.target_cntr_net.load_state_dict(self.policy_cntr_net.state_dict())
        self.target_cntr_net.eval()
        self.cntr_optimizer_name, self.cntr_loss_name, self.cntr_lr_name = cntr_optimizer, cntr_loss, cntr_lr
        self.cntr_optimizer, self.cntr_criterion = self.set_optim(self.policy_cntr_net.parameters(), 
            self.cntr_optimizer_name, self.cntr_loss_name, self.cntr_lr_name)
        self.cntr_clamp = cntr_clamp

    def set_optim(self, params, optimizer_str, loss_str, lr):
        if optimizer_str == 'SGD':
            optimizer = optim.SGD(params, lr=lr)
        if optimizer_str == 'Adam':
            optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
                amsgrad=False)
        if optimizer_str == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, 
                momentum=0, centered=False)

        if loss_str == 'MSEloss':
            criterion = nn.MSELoss()
        if loss_str == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()

        return optimizer, criterion


    def select_goal(self, agent_state):
        # softmax approach
        with torch.no_grad():
            Q = np.zeros(len(self.env.current_objects))
            for counter, goal in enumerate(self.env.current_objects):
                agent_env_state = utils.agent_env_state(agent_state, self.env.env_state)
                # goal_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(goal),0),0)
                pred = self.Q_meta(agent_env_state, goal, False)
                Q[counter] = pred.item()

            goal_idxs = np.arange(len(Q))
            probabilities = np.exp(self.meta_policy_temp*Q) / sum(np.exp(self.meta_policy_temp * Q))
            goal_idx = np.random.choice(goal_idxs, 1, p=probabilities)[0]
            goal = self.env.current_objects[goal_idx]

        # update environment
        self.env.selected_goals.append(goal)
        return goal

    # def select_goal(self, agent_state):
    #     # epsilon greedy
    #     if self.meta_policy_temp < random.random():
    #         with torch.no_grad():
    #             Q = []
    #             for goal in self.env.current_objects:
    #                 agent_env_state = utils.agent_env_state(agent_state, self.env.env_state)
    #                 pred = self.Q_meta(agent_env_state, goal, False)
    #                 Q.append(pred.item())
    #             goal_idx = np.argmax(Q)
    #             goal = self.env.current_objects[goal_idx]
    #     else:
    #         print("Exploring ...............")
    #         goal = self.random_goal_selection()
    #     # update environment
    #     self.env.selected_goals.append(goal)
    #     return goal

    def random_goal_selection(self):
        # Don't call this function directly, it would always be called from select_goal()
        goal_idx = int(np.random.choice(len(self.env.current_objects)))
        goal = self.env.current_objects[goal_idx]
        return goal

    def select_action(self, agent_state, env_state, goal):
        # softmax policy
        i = agent_state[0,0].item()
        j = agent_state[0,1].item()
        with torch.no_grad():
            agent_env_state = utils.agent_env_state(agent_state, env_state)
            action_probs_tensor = self.Q_cntr(agent_env_state, goal, target=False)
            action_probs = np.squeeze(action_probs_tensor.numpy())
            action_idxs = np.arange(len(action_probs))
            probabilities = np.exp(self.cntr_policy_temp*action_probs) / np.sum(np.exp(self.cntr_policy_temp*action_probs))
            action_idx = torch.tensor(np.array(np.random.choice(action_idxs, 1, p=probabilities)[0]))
            action = self.env.all_actions[action_idx.item()]
        return action_idx, action


    # def select_action(self, agent_state, env_state, goal):
    #     # epsilon greedy
    #     i = agent_state[0,0].item()
    #     j = agent_state[0,1].item()
    #     if random.random() > self.cntr_policy_temp:
    #         with torch.no_grad():
    #             agent_env_state = utils.agent_env_state(agent_state, env_state)
    #             action_probs = self.Q_cntr(agent_env_state, goal, target=False) 
    #             action_idx = action_probs.max(1)[1]
    #             action = self.env.all_actions[action_idx.item()]
    #     else:
    #         action_idx, action = self.random_action_selection()
    #     return action_idx, action

    def random_action_selection(self):
        # print("random action selected")
        # i = self.env.agent_loc[0,0].item() 
        # j = self.env.agent_loc[0,1].item()
        # allowable_action_idxs = self.env.allowable_action_idxs[i,j]
        all_actions = self.env.all_actions
        action_idx = int(np.random.choice(np.arange(4)))
        action = self.env.all_actions[action_idx]
        return action_idx, action

    def store(self, *args, meta):
        if meta:
            self.meta_memory.push(*args)
            # if not self.meta_memory.memory[-1].terminal:
        else:
            self.cntr_memory.push(*args)


    def Q_cntr(self, agent_env_state, goal, target):
        if target:
            try:
                Q_policys = self.target_cntr_net(agent_env_state, goal) 
            except Exception as e:
                agent_env_state = torch.unsqueeze(agent_env_state, dim=0)
                goal = torch.unsqueeze(torch.tensor([goal], dtype=torch.float32, device=device), dim=0)
                Q_policys = self.target_cntr_net(agent_env_state, goal) 
        else:
            try:
                Q_policys = self.policy_cntr_net(agent_env_state, goal) 
            except Exception as e:
                agent_env_state = torch.unsqueeze(agent_env_state, dim=0)
                goal = torch.unsqueeze(torch.tensor([goal], dtype=torch.float32, device=device), dim=0)
                Q_policys = self.policy_cntr_net(agent_env_state, goal)
        return Q_policys

    def Q_meta(self, agent_env_state, goal, target):
        if target:
            try:
                Q_policys = self.target_meta_net(agent_env_state, goal)
            except Exception as e:
                agent_env_state = torch.unsqueeze(agent_env_state, dim=0)
                goal = torch.unsqueeze(torch.tensor([goal], dtype=torch.float32, device=device), dim=0)
                Q_policys = self.target_meta_net(agent_env_state, goal)
        else:
            try:
                Q_policys = self.policy_meta_net(agent_env_state, goal)
            except Exception as e:
                agent_env_state = torch.unsqueeze(agent_env_state, dim=0)
                goal = torch.unsqueeze(torch.tensor([goal], dtype=torch.float32, device=device), dim=0)
                Q_policys = self.policy_meta_net(agent_env_state, goal)
        return Q_policys


    def _update_cntr(self):
        if len(self.cntr_memory) < self.batch_size:
            return 100000
        sample_size = min(self.batch_size, len(self.cntr_memory))
        exps = self.cntr_memory.sample(sample_size)

        
        state_batch = torch.cat([torch.unsqueeze(exp.agent_env_cntr, 0) for exp in exps])
        meta_goal_batch = torch.cat([torch.unsqueeze(exp.meta_goal,0) 
            for exp in exps])
        not_cntr_done_mask = torch.tensor(tuple(map(lambda s: s != True, [exp.cntr_done for exp in exps])), 
            device=device)
        
        
        # calculating V at the next state
        next_state_Vs = torch.zeros(sample_size, device=device)
        all_cntr_done = False
        try:
            next_state_non_cntr_dones = torch.cat([torch.unsqueeze(exp.next_agent_env_cntr, 0) 
                                            for exp in exps if exp.cntr_done != True])
        except:
            all_cntr_done = True
        if not all_cntr_done:
            next_state_Vs[not_cntr_done_mask] = self.Q_cntr(next_state_non_cntr_dones, 
                    meta_goal_batch[not_cntr_done_mask], target=True).max(1)[0].detach()

        # action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor([exp.int_reward for exp in exps], dtype=torch.float32, device=device).detach()
        action_batch = torch.unsqueeze(torch.tensor([exp.action_idx for exp in exps], device=device), 1).detach()
        Q_policys = self.Q_cntr(state_batch, meta_goal_batch, target=False).gather(1, action_batch)

        try:
            Q_targets = reward_batch + (next_state_Vs * self.gamma)
        except:
            print("Q targets problems...")

        Q_targets = torch.unsqueeze(Q_targets, 1)
        self.cntr_optimizer.zero_grad()
        loss = self.cntr_criterion(Q_policys, Q_targets)
        loss.backward()
        if self.cntr_clamp:
            for param in self.policy_cntr_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.cntr_optimizer.step()
        
        #Update target network
        with torch.no_grad():
            cntr_weights = self.policy_cntr_net.parameters()
            cntr_target_weights = self.target_cntr_net.parameters()

            for layer_w, target_layer_w in zip(cntr_weights, cntr_target_weights):
                target_layer_w = self.target_tau * layer_w + (1 - self.target_tau) * target_layer_w

        return loss.item()

    def _update_meta(self):
        # ["agent_env_state_0", "goal", "reward", "next_agent_env_state", "next_available_goals", "done"]

        if len(self.meta_memory) < self.meta_batch_size:
            return 100000


        sample_size = min(self.meta_batch_size, len(self.meta_memory))
        exps = self.meta_memory.sample(sample_size)
        # exps = self.cntr_Transition(*zip(*exps))

        state_batch = torch.cat([torch.unsqueeze(exp.agent_env_state_0, 0) for exp in exps])
        meta_goal_batch = torch.cat([torch.unsqueeze(exp.meta_goal,0) for exp in exps])
        Q_policys = self.Q_meta(state_batch, meta_goal_batch, target=False)
        reward_batch = torch.tensor([exp.reward for exp in exps], dtype=torch.float32, device=device)

        # non_terminal_mask = torch.tensor(tuple(map(lambda s: s != True, [exp.terminal for exp in exps])))
        # try:
        #     next_state_non_terminals = torch.cat([torch.unsqueeze(utils.agent_env_state2(n_dim, 
        #         exp.next_agent_env_state, exp.goal), 0) for exp in exps if exp.terminal != True])
        #     next_state_Vs = self.Q_meta(next_state_non_terminals, target=True)[0].detach()
        # except:
        #     pass


        Q_targets = torch.zeros((sample_size, 1), device=device)

        next_state_Vs = torch.zeros(sample_size, device=device)
        # next_state_Vs[non_terminal_mask] = self.Q_cntr(next_state_non_terminals, 
        #     target=True)[0].max(1)[0].detach()
        # Q_targets = reward_batcht + (next_state_Vs * self.gamma) 
        
        Q_targets[:,0] = reward_batch

        for i, exp in enumerate(exps):
            if not exp.terminal:
                # this block finds the max Q in the next state for this particular experiment
                # if exp.terminal is true, it means that the next state is terminal and we have 
                # Q(s,.)=0 if s is terminal 
                
                next_goals = torch.cat([torch.unsqueeze(torch.tensor([next_goal], device=device), 0)
                                for next_goal in exp.next_available_goals])
                next_agent_env_states = torch.cat([torch.unsqueeze(exp.next_agent_env_state, 0)
                                for next_goal in next_goals])
                
                next_state_V = max(self.Q_meta(next_agent_env_states, next_goals, target=True)).item()
                Q_targets[i,0] +=  self.gamma * (next_state_V)
                # Q_targets = (next_state_Vs * self.gamma) + reward_batch



        self.meta_optimizer.zero_grad()
        loss = self.meta_criterion(Q_policys, Q_targets)
        loss.backward()
        if self.meta_clamp:
            for param in self.policy_meta_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()

        #Update target network
        with torch.no_grad():
            meta_weights = self.policy_meta_net.parameters()
            meta_target_weights = self.target_meta_net.parameters()

            for layer_w, target_layer_w in zip(meta_weights, meta_target_weights):
                target_layer_w = self.target_tau * layer_w + (1 - self.target_tau) * target_layer_w
        return loss.item()

    def update(self, meta=False):
        if meta:
            loss = self._update_meta()
        else:
            loss = self._update_cntr()
        return loss