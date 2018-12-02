# used part of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop
import sys
sys.path.append('../')
from envs.gridworld_relational1 import Gridworld
from pdb import set_trace

# DEFAULT ARCHITECTURE FOR THE META CONTROLLER
default_meta_input_dim = 83
default_meta_layers = [Dense] * 5
default_meta_inits = ['lecun_uniform'] * 5
default_meta_nodes = [20, 30, 30, 30, 10] 
default_meta_activations = ['relu'] * 5
default_meta_loss = "mean_squared_error"
default_meta_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_meta_batch_size = 1000
default_meta_epsilon = 1.0

# DEFAULT ARCHITECTURES FOR THE LOWER LEVEL CONTROLLER/controller
default_input_dim = 84
default_layers = [Dense] * 5
default_inits = ['lecun_uniform'] * 5
default_nodes = [20, 30, 30, 30, 4] 
default_activations = ['relu'] * 5
default_loss = "mean_squared_error"
default_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_batch_size = 1000
default_gamma = 0.975
default_epsilon = 1.0
default_tau = 0.001


class hDQN:

    def __init__(self, 
                env, 
                meta_layers=default_meta_layers,
                meta_inits=default_meta_inits,
                meta_nodes=default_meta_nodes, 
                meta_activations=default_meta_activations,
                meta_input_dim = default_meta_input_dim,
                meta_loss=default_meta_loss, 
                meta_optimizer=default_meta_optimizer,
                layers=default_layers, 
                inits=default_inits, 
                nodes=default_nodes,
                activations=default_activations, 
                input_dim = default_input_dim,
                loss=default_loss,
                optimizer=default_optimizer, 
                batch_size=default_batch_size,
                meta_batch_size=default_meta_batch_size, 
                gamma=default_gamma,
                meta_epsilon=default_meta_epsilon, 
                epsilon=default_epsilon, 
                tau = default_tau):
        self.env = env
        self.meta_layers = meta_layers
        self.meta_inits = meta_inits
        self.meta_nodes = meta_nodes
        self.meta_activations = meta_activations
        self.meta_input_dim = meta_input_dim
        self.meta_loss = meta_loss
        self.meta_optimizer = meta_optimizer
        self.layers = layers
        self.inits = inits
        self.nodes = nodes
        self.activations = activations
        self.input_dim = input_dim
        self.loss = loss
        self.optimizer = optimizer
        self.meta_controller = self.meta_controller()
        self.target_meta_controller = self.target_meta_controller()
        self.controller = self.controller()
        self.target_controller = self.target_controller()
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


    def meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](units=self.meta_nodes[0] , kernel_initializer=self.meta_inits[0], 
            input_dim=self.meta_input_dim, activation=self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, kernel_initializer=init, input_shape=(node,), activation=activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta
    
    def target_meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](units=self.meta_nodes[0] , kernel_initializer=self.meta_inits[0], 
            input_dim=self.meta_input_dim, activation=self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, kernel_initializer=init, input_shape=(node,), activation=activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta

    def controller(self):
        controller = Sequential()
        controller.add(self.layers[0](self.nodes[0], kernel_initializer=self.inits[0], 
            input_dim=self.input_dim, activation=self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            controller.add(layer(node, kernel_initializer=self.inits[0], activation=activation))
        controller.compile(loss=self.loss, optimizer=self.optimizer)
        return controller
    
    def target_controller(self):
        controller = Sequential()
        controller.add(self.layers[0](self.nodes[0], kernel_initializer=self.inits[0], 
            input_dim=self.input_dim, activation=self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            controller.add(layer(node, kernel_initializer=self.inits[0], activation=activation))
        controller.compile(loss=self.loss, optimizer=self.optimizer)
        return controller

    def select_goal(self, agent_state):
        if self.meta_epsilon < random.random():
            Q = []
            for goal in self.env.current_objects:
                state_goal = np.concatenate((self.env.grid_flat, agent_state, 
                                             np.array(goal).reshape((1,1))), axis=1)
            
                pred = self.meta_controller.predict(state_goal, verbose=0)
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


    def select_action(self, agent_state, goal):
        agent_env_state = np.concatenate((self.env.grid_flat, agent_state), axis=1)
        state_goal_feature = np.concatenate((agent_env_state, np.array(goal).reshape((1,1))), axis=1)
        if random.random() > self.epsilon:
            # print("controller selected action")
            # ensures that only actions that cause movement are chosen
            action_probs = self.controller.predict(state_goal_feature, verbose=0)        
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
            if len(self.meta_memory) > 1000000:
                self.meta_memory = self.meta_memory[-1000:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 1000000:
                self.memory = self.memory[-1000000:]

    def Q_controller(self, state_vectors, target):
        if target:
            try:
                Q_preds = self.target_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                Q_preds = self.target_controller.predict(state_vectors, verbose=0)
            return Q_preds, state_vectors
        else:
            try:
                Q_preds = self.controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                Q_preds = self.controller.predict(state_vectors, verbose=0)
            return Q_preds, state_vectors

    def Q_meta(self, state_vectors, target):
        if target:
            try:
                Q_preds = self.target_meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                Q_preds = self.target_meta_controller.predict(state_vectors, verbose=0)
            return Q_preds, state_vectors
        else:
            try:
                Q_preds = self.meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                Q_preds = self.meta_controller.predict(state_vectors, verbose=0)
            return Q_preds, state_vectors

    def _update_controller(self):
        sample_size = min(self.batch_size, len(self.memory))
        exps = [random.choice(self.memory) for _ in range(sample_size)]
        state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, 
                        exp.goal], axis=1) for exp in exps]))
        next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, 
                        exp.goal], axis=1) for exp in exps]))

        Q_targets, state_vectors  = self.Q_controller(state_vectors, target=False)
        next_state_Q_preds, next_state_vectors = self.Q_controller(next_state_vectors, target=True)

        for i, exp in enumerate(exps):
            Q_targets[i,exp.action] = exp.reward
            # if not exp.controller_done:
            Q_targets[i,exp.action] += self.gamma * max(next_state_Q_preds[i])
        # Q_preds = np.asarray(Q_preds)
        self.controller.fit(state_vectors, Q_targets, verbose=0)
        

        #Update target network
        controller_weights = self.controller.get_weights()
        controller_target_weights = self.target_controller.get_weights()
        for i in range(len(controller_weights)):
            controller_target_weights[i] = self.target_tau * controller_weights[i] + (1 - self.target_tau) * controller_target_weights[i]
        self.target_controller.set_weights(controller_target_weights)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            sample_size = min(self.meta_batch_size, len(self.meta_memory))
            exps = [random.choice(self.meta_memory) for _ in range(sample_size)]
            state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], 
                                    axis=1) for exp in exps]))

            Q_preds, state_vectors = self.Q_meta(state_vectors, target=False)
            Q_targets = np.zeros_like(Q_preds, dtype=int)
            for i, exp in enumerate(exps):
                Q_targets[i,0] = exp.reward

                # if not exp.done:
                if exp.next_available_goals:
                    # this block finds the max Q in the next state for this particular experiment
                    next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, next_goal.reshape((1,1))], 
                                    axis=1) for next_goal in exp.next_available_goals]))
                    next_state_Q_preds, next_state_vectors = self.Q_meta(next_state_vectors, target=True)
                    Q_targets[i,0] += self.gamma * max(next_state_Q_preds)

            self.meta_controller.fit(state_vectors, Q_targets, verbose=0)
            
            #Update target network
            meta_weights = self.meta_controller.get_weights()
            meta_target_weights = self.target_meta_controller.get_weights()
            for i in range(len(meta_weights)):
                meta_target_weights[i] = self.target_tau * meta_weights[i] + (1 - self.target_tau) * meta_target_weights[i]
            self.target_meta_controller.set_weights(meta_target_weights)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update_controller()