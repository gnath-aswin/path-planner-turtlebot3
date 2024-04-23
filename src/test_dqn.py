#!/usr/bin/env python

import torch as T
import rospy
from dqn import DeepQNetwork
from environment import Env


if __name__ == '__main__':
    done = False
    rospy.init_node("test",anonymous=True)
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        

    actor = DeepQNetwork(lr=0.001,n_actions=5,input_dims=[12],fc1_dims=256,fc2_dims=256)
    checkpoint = T.load('models/dqn/epoch6000_ 2.pth')
    actor.load_state_dict(checkpoint['model_state_dict'])

    env = Env()
    state = env.reset()
    actions = actor(T.from_numpy(state).to(device))
    while not done:
        action = T.argmax(actions).item()
        observation_,reward,done,info = env.step(action)
        if info:
            done = True
        actions = actor(T.from_numpy(observation_).to(device))
        

