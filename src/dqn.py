#!/usr/bin/env python

import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) class for reinforcement learning.

    Attributes:
        input_dims (tuple): Dimensions of the input state.
        fc1_dims (int): Number of units in the first fully connected layer.
        fc2_dims (int): Number of units in the second fully connected layer.
        n_actions (int): Number of actions in the action space.
    """
    def __init__(self, lr: float, input_dims: tuple, fc1_dims: int, fc2_dims: int, n_actions: int):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """
        Perform forward pass through the network.

        Args:
            state (Tensor): Input state tensor.

        Returns:
            Tensor: Output actions tensor.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.output(x)
        return actions

