import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define experience tuple type
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class RLAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
        
    def get_state(self, camera_pos: Tuple[float, float], target_pos: Tuple[float, float], 
                 obstacles: List[Tuple[float, float]], visited_pois: List[bool]) -> torch.Tensor:
        """Convert environment information into state tensor"""
        state = []
        
        # Add camera position
        state.extend([camera_pos[0], camera_pos[1]])
        
        # Add target position
        state.extend([target_pos[0], target_pos[1]])
        
        # Add distance and angle to target
        dx = target_pos[0] - camera_pos[0]
        dy = target_pos[1] - camera_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        state.extend([distance, angle])
        
        # Add nearest obstacle information (up to 5 nearest obstacles)
        distances = []
        for obs in obstacles:
            dx = obs[0] - camera_pos[0]
            dy = obs[1] - camera_pos[1]
            distances.append((np.sqrt(dx**2 + dy**2), dx, dy))
        
        distances.sort()  # Sort by distance
        for i in range(min(5, len(distances))):
            state.extend([distances[i][1], distances[i][2]])  # Add dx, dy for each obstacle
        
        # Pad with zeros if less than 5 obstacles
        while len(distances) < 5:
            state.extend([0.0, 0.0])
            
        # Add visited POIs status
        state.extend([1.0 if v else 0.0 for v in visited_pois])
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_tensor
        
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy with bias towards movement"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            # Get Q-values
            q_values = self.policy_net(state)
            logger.info(f"Q-values: {q_values.squeeze().tolist()}")
            
            # Select action
            if random.random() > self.epsilon:
                # Add bias against no-movement action
                q_values_adj = q_values.clone()
                q_values_adj[0, 3] -= 0.2  # Penalize no-movement action
                action = q_values_adj.max(1)[1].item()
                logger.info(f"Selected greedy action {action}")
            else:
                # During exploration, prefer movement actions
                if random.random() < 0.8:  # 80% chance to choose movement
                    action = random.randint(0, 2)  # Only choose from movement actions
                else:
                    action = random.randrange(self.action_size)
                logger.info(f"Selected random action {action}")
            
            return action
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def update_model(self) -> Optional[float]:
        """Update model weights using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch from memory
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors with float32 dtype
        state_batch = torch.cat(batch.state).to(dtype=torch.float32)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.int64)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.cat(batch.next_state).to(dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Compute loss and update weights
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        logger.info(f"Model loaded from {path}")
