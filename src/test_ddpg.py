#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from ddpg_tf import Agent
from environment import Env
import rospy
from networks import ActorNetwork


if __name__ == '__main__':
    rospy.init_node('env', anonymous=True)
    # Initialize environment
    env = Env()
    # Initialize agent
    agent = Agent(input_dims=[12], n_actions=1)
    evaluate = True


    actor = ActorNetwork(n_actions=1)
    dummy_observation = np.zeros((1, 12))  # Example dummy input data
    _ = actor(dummy_observation)
    actor.load_weights('models/ddpg/obstacle/pos_reward/actor_ddpg.h5')



    observation = env.reset()
    done, info = False, False
    score = 0
    while not done:
        if done:
            break
        if info:
            break
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action = actor(state)            
        observation_, reward, done, info = env.step(*action)
        score += reward
        observation = observation_

print(f'Score: {score}\nReward: {reward}')