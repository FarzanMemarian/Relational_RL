# used part of the code from https://github.com/EthanMacdonald/h-DQN/blob/master/agent/hDQN.py

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop
import sys
sys.path.append('../')
from envs.gridworld_relational import Gridworld
from pdb import set_trace

# DEFAULT ARCHITECTURE FOR THE META CONTROLLER
default_meta_layers = [Dense] * 5
default_meta_inits = ['lecun_uniform'] * 5
default_meta_nodes = [20, 30, 30, 30, 10] 
default_meta_input_dim = 83
                # 83 = number of squares(81) + agent_state(2)
                # 10 is the number of objects
default_meta_activations = ['relu'] * 5
default_meta_loss = "mean_squared_error"
default_meta_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_meta_batch_size = 1000
default_meta_epsilon = 1.0

# DEFAULT ARCHITECTURES FOR THE LOWER LEVEL CONTROLLER/ACTOR
default_layers = [Dense] * 5
default_inits = ['lecun_uniform'] * 5
default_nodes = [20, 30, 30, 30, 4] 
default_input_dim = 84
                # 84 = number of squares(81) + agent_state(2) + goal(1)
                # 4 is the total number of moves
default_activations = ['relu'] * 5
default_loss = "mean_squared_error"
default_optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
default_batch_size = 1000
default_gamma = 0.975
default_epsilon = 1.0
default_actor_epsilon = 1.0
default_tau = 0.001

# Default architecture for the meta controller
# default_meta_layers = 

class hDQN:

    def __init__(self, env, meta_layers=default_meta_layers, meta_inits=default_meta_inits,
                meta_nodes=default_meta_nodes, meta_activations=default_meta_activations,
                meta_input_dim = default_meta_input_dim,
                meta_loss=default_meta_loss, meta_optimizer=default_meta_optimizer,
                layers=default_layers, inits=default_inits, nodes=default_nodes,
                activations=default_activations, input_dim = default_input_dim,
                loss=default_loss,
                optimizer=default_optimizer, batch_size=default_batch_size,
                meta_batch_size=default_meta_batch_size, gamma=default_gamma,
                meta_epsilon=default_meta_epsilon, epsilon=default_epsilon, 
                actor_epsilon = default_actor_epsilon, tau = default_tau,
                ):
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
        self.actor = self.actor()
        self.target_actor = self.target_actor()
        self.goal_selected = np.zeros(len(self.env.original_objects))
        self.goal_success = np.zeros(len(self.env.original_objects))
        self.meta_epsilon = meta_epsilon
        self.actor_epsilon = actor_epsilon
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.gamma = gamma
        self.target_tau = tau
        self.memory = []
        self.meta_memory = []


    def meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](self.meta_nodes[0] , init=self.meta_inits[0], input_dim=self.meta_input_dim))
        meta.add(Activation(self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, init=init, input_shape=(node,)))
            meta.add(Activation(activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta
    
    def target_meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](self.meta_nodes[0] , init=self.meta_inits[0], input_dim=self.meta_input_dim))
        meta.add(Activation(self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, init=init, input_shape=(node,)))
            meta.add(Activation(activation))
            print("meta node: " + str(node))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta

    def actor(self):
        actor = Sequential()
        actor.add(self.layers[0](self.nodes[0], kernel_initializer=self.inits[0], input_dim=self.input_dim))
        actor.add(Activation(self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            print(node)
            actor.add(layer(node, kernel_initializer=self.inits[0]))
            actor.add(Activation(activation))
        actor.compile(loss=self.loss, optimizer=self.optimizer)
        return actor
    
    def target_actor(self):
        actor = Sequential()
        actor.add(self.layers[0](self.nodes[0], kernel_initializer=self.inits[0], input_dim=self.input_dim))
        actor.add(Activation(self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            print(node)
            actor.add(layer(node, kernel_initializer=self.inits[0]))
            actor.add(Activation(activation))
        actor.compile(loss=self.loss, optimizer=self.optimizer)
        return actor
        


    def select_goal(self, agent_state):
        agent_env_state = np.concatenate((self.env.grid_flattened, agent_state.reshape((1,2))), axis=1)
        if self.meta_epsilon > random.random():
            pred = self.meta_controller.predict(agent_env_state, verbose=0)
            print("pred shape: " + str(pred.shape))
            goal_idx = np.argmax(pred)
            goal = self.env.original_objects[goal_idx]
        else:
            print("Exploring")
            goal_idx, goal = self.random_goal_selection()
        self.env.update_target_goal()
        self.env.selected_goals.append(goal)
        return goal_idx, goal

    def random_goal_selection(self):
        goal_idx = np.random.choice(len(self.env.original_objects))
        goal = self.env.original_objects[goal_idx]
        self.env.selected_goals.append(goal)
        return goal_idx, goal

    def select_action(self, agent_state, goal, goal_idx):
        agent_env_state = np.concatenate((self.env.grid_flattened, agent_state.reshape((1,2))), axis=1)
        state_goal_feature = np.concatenate((agent_env_state, np.array(goal).reshape((1,1))), axis=1)
        if random.random() > self.actor_epsilon:
            action_idx = np.argmax(self.actor.predict(state_goal_feature, verbose=0))
            action = self.env.all_actions[action_idx]
        else:
            action_idx, action = self.random_move_selection()
        return action_idx, action


    def random_move_selection(self):
        action_idx = np.random.choice(len(self.env.allowable_actions[self.env.i, self.env.j]))
        action = self.env.all_actions[action_idx]
        return action_idx, action


    # def criticize(self, goal, next_agent_state):
    #     return 1.0 if goal == self.env.grid_mat[next_agent_state[0], next_agent_state[1]] else 0.0

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
            if len(self.meta_memory) > 1000000:
                self.meta_memory = self.meta_memory[-100:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 1000000:
                self.memory = self.memory[-1000000:]

    def _update_controller(self):
        sample_size = min(self.batch_size, len(self.memory))
        exps = [random.choice(self.memory) for _ in range(sample_size)]
        state_vectors = np.squeeze(np.asarray([np.concatenate([self.env.grid_flattened, 
                        exp.agent_state.reshape((1,2)), exp.goal.reshape((1,1))], axis=1) for exp in exps]))
        next_state_vectors = np.squeeze(np.asarray([np.concatenate([self.env.grid_flattened, 
                        exp.next_agent_state.reshape((1,2)), exp.goal.reshape((1,1))], axis=1) for exp in exps]))
        try:
            Q_preds = self.actor.predict(state_vectors, verbose=0)
        except Exception as e:
            state_vectors = np.expand_dims(state_vectors, axis=0)
            Q_preds = self.actor.predict(state_vectors, verbose=0)
        
        try:
            next_state_Q_preds = self.target_actor.predict(next_state_vectors, verbose=0)
        except Exception as e:
            next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
            next_state_Q_preds = self.target_actor.predict(next_state_vectors, verbose=0)
        
        for i, exp in enumerate(exps):
            Q_preds[i,exp.action] = exp.reward
            if not exp.done:
                Q_preds[i,exp.action] += self.gamma * max(next_state_Q_preds[i])
        # Q_preds = np.asarray(Q_preds)
        self.actor.fit(state_vectors, Q_preds, verbose=0)
        
        #Update target network
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.target_tau * actor_weights[i] + (1 - self.target_tau) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            sample_size = min(self.meta_batch_size, len(self.meta_memory))
            exps = [random.choice(self.meta_memory) for _ in range(sample_size)]
            state_vectors = np.squeeze(np.asarray([np.concatenate([self.env.grid_flattened, 
                          exp.agent_state.reshape((1,2))], axis=1) for exp in exps]))
            next_state_vectors = np.squeeze(np.asarray([np.concatenate([self.env.grid_flattened, 
                               exp.next_agent_state.reshape((1,2))], axis=1) for exp in exps]))
            try:
                Q_preds = self.meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                Q_preds = self.meta_controller.predict(state_vectors, verbose=0)
            
            try:
                next_state_Q_preds = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            except Exception as e:
                next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
                next_state_Q_preds = self.target_meta_controller.predict(next_state_vectors, verbose=0)
            
            for i, exp in enumerate(exps):
                Q_preds[i,np.argmax(exp.goal)] = exp.reward
                if not exp.done:
                    Q_preds[i,np.argmax(exp.goal)] += self.gamma * max(next_state_Q_preds[i])
            self.meta_controller.fit(state_vectors, Q_preds, verbose=0)
            
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