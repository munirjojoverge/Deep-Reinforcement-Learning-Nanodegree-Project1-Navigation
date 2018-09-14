#****************************************************
# Deep Reinforcement Learning Nano-degree - Udacity
#            Created on: September 12, 2018
#                Author: Munir Jojo-Verge
#****************************************************

import numpy as np
import random
from collections import namedtuple, deque

from model import DQN
from model import DQN_NoisyNet

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, args, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            args (class defined on the notebook): A set of parameters that will define the agent hyperparameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.params = args

        # Deep Q-Network
        if args.use_NoisyNet:
            self.DQN_local = DQN_NoisyNet(args, state_size, action_size).to(args.device)
            self.DQN_target = DQN_NoisyNet(args, state_size, action_size).to(args.device)
        else:
            self.DQN_local = DQN(args, state_size, action_size).to(args.device)
            self.DQN_target = DQN(args, state_size, action_size).to(args.device)

        self.optimizer = optim.Adam(self.DQN_local.parameters(), lr=args.lr, eps=args.adam_eps)

        # Replay memory
        self.memory = ReplayBuffer(args, action_size)
        # Initialize time step (for updating every args.target_update steps)
        self.t_step = 0

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.DQN_local.reset_noise()
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every args.target_update time steps.
        self.t_step = (self.t_step + 1) % self.params.target_update
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.params.device)
        self.DQN_local.eval()
        with torch.no_grad():
            action_values = self.DQN_local(state)
        self.DQN_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            args.discount (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.DQN_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.params.discount * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.DQN_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.DQN_local, self.DQN_target, self.params.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            args.tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, args, action_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            args.memory_capacity (int): maximum size of buffer
            args.batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=args.memory_capacity)  
        self.batch_size = args.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(args.seed)
        self.device = args.device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k= self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)