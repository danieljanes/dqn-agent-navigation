from collections import deque
from collections import namedtuple
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing_extensions import Final

from navigation.model import DQN


CAPACITY: Final = int(1e5)
BATCH_SIZE: Final = 64
GAMMA: Final = 0.99
TAU: Final = 1e-3
LEARNING_RATE: Final = 5e-4
UPDATE_EVERY: Final = 1
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOUBLE_Q_LEARNING: Final = True


class Agent:

    def __init__(self, state_size, action_size, seed = None):
        # dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # DQN
        self.dqn_policy = DQN(state_size, action_size, seed).to(DEVICE)
        self.dqn_target = DQN(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.dqn_policy.parameters(), lr=LEARNING_RATE)
        # Replay Memory
        self.memory = ReplayBuffer(action_size, CAPACITY, BATCH_SIZE, seed)
        self.t_step = 0  # Initialize time step (for updating every UPDATE_EVERY steps)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Improve policy every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough examples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self._learn(experiences, GAMMA)

    def act(self, state, epsilon=0.0) -> int:
        """
        Takes a state, returns an action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.dqn_policy.eval()
        with torch.no_grad():
            action_values = self.dqn_policy(state)
        self.dqn_policy.train()
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences: List, gamma: float) -> None:
        states, actions, rewards, next_states, dones = experiences
        
        dones_bytes = dones.type(torch.uint8)        

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.dqn_policy(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states
        if DOUBLE_Q_LEARNING:
            # Pick action using Policy Network
            next_state_predictions = self.dqn_policy(next_states)
            next_state_actions = next_state_predictions.argmax(1)
            # Evaluate action value using Target Network
            next_state_action_values = self.dqn_target(next_states)
            next_state_values_unsqueezed = next_state_action_values.gather(1, next_state_actions.view(-1, 1))
            next_state_values = next_state_values_unsqueezed.squeeze().detach()
        else:
            # Pick and evaluate action using Target Network
            next_state_predictions = self.dqn_target(next_states)
            next_state_max_prediction = next_state_predictions.max(1)
            next_state_values = next_state_max_prediction[0].detach()        
        next_state_values[dones_bytes.squeeze()] = 0

        # Compute the expected Q values ("target" for calculating the loss)
        expected_state_action_values = rewards.squeeze() + (next_state_values * GAMMA)

        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.dqn_policy, self.dqn_target, TAU)

    def _soft_update(self, dqn_policy, dqn_target, tau: float) -> None:
        """
        Soft update model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            dqn_policy (PyTorch model): weights will be copied from
            dqn_target (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, policy_param in zip(dqn_target.parameters(), dqn_policy.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Source: https://github.com/udacity/deep-reinforcement-learning
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
