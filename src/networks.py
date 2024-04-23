import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 




class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, name='critic', checkpoint_dir="tmp"):
        super(CriticNetwork, self).__init__()
        self.fc1 = fc1_dims
        self.fc2 = fc2_dims

        self.model_name  = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1, activation='relu')
        self.fc2 = Dense(self.fc2, activation='relu') 
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q
    

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name='actor', checkpoint_dir="tmp"):
        super(ActorNetwork, self).__init__()
        self.fc1 = fc1_dims
        self.fc2 = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1, activation='relu')
        self.fc2 = Dense(self.fc2, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')# activation fn based on action space of env,scale accordingly

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        # If action bound is not +/- 1, scale here accordingly
        mu = self.mu(prob)
        return mu*1.5 # scale to action bound of env