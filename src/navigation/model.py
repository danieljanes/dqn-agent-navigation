import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        # layers
        intermediate_units = 128
        self.fc1 = nn.Linear(state_size, intermediate_units)
        self.fc2 = nn.Linear(intermediate_units, intermediate_units)
        self.fc3 = nn.Linear(intermediate_units, intermediate_units)
        self.fc4 = nn.Linear(intermediate_units, intermediate_units)
        self.fc_n = nn.Linear(intermediate_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_n(x)
        return x
