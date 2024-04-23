import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, 
                 gamma=0.99, n_actions=1, max_size=1000000, tau=0.005, 
                 fc1_dims=256, fc2_dims=256, batch_size=256, noise=0.3):
        self.input_dims = input_dims
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = 1.5
        self.min_action = -1.5

        self.actor = ActorNetwork(n_actions, name='actor')
        self.critic = CriticNetwork(fc1_dims, fc2_dims, name='critic')
        self.target_actor = ActorNetwork(n_actions, fc1_dims, fc2_dims, name='target_actor')
        self.target_critic = CriticNetwork(fc1_dims, fc2_dims, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))#no optim but required to compile
        self.target_critic.compile(optimizer=Adam(learning_rate=beta)) #no optim but required to compile

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)



    def choose_action(self, observation, evaluation=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluation:
            actions += tf.random.uniform(shape=[self.n_actions], minval=0.0-self.noise, maxval=0.0+self.noise, dtype=tf.float32)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions.numpy()[0]
   


        