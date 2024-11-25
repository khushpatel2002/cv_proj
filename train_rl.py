import numpy as np
import torch
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from dev import Environment, GRID_SIZE
from rl_agent import RLAgent
import tensorboard
from torch.utils.tensorboard import SummaryWriter



# Constants for training
EPISODES = 5000
MAX_STEPS = 500
SAVE_INTERVAL = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000

# Action space:
# 0: Move forward
# 1: Turn left
# 2: Turn right
# 3: No movement
NUM_ACTIONS = 4

def calculate_reward(env, prev_distance, current_distance, reached_poi, collision):
    """Calculate reward based on various factors"""
    reward = 0
    
    # Larger reward for getting closer to target
    distance_reward = prev_distance - current_distance
    reward += distance_reward * 0.5
    
    # Bigger reward for reaching POI
    if reached_poi:
        reward += 20.0
    
    # Reduced collision penalty
    if collision:
        reward -= 2.0
    
    # Smaller step penalty
    reward -= 0.005
    
    return reward

def get_obstacles_from_grid(env):
    """Get obstacle positions from the environment grid"""
    obstacles = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if env.grid[i,j] == 1:  # If there's a wall
                obstacles.append((i, j))
    return obstacles

def train():
    # Create directories for saving models and logs
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/camera_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Initialize environment and agent
    env = Environment("custom_environment.json")
    
    # Calculate state size based on our state representation
    # [camera_x, camera_y, target_x, target_y, distance, angle, 
    #  5 nearest obstacles (dx,dy), visited_pois status]
    state_size = 2 + 2 + 2 + (5 * 2) + len(env.points_of_interest)
    
    agent = RLAgent(state_size=state_size, action_size=NUM_ACTIONS)
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in tqdm(range(EPISODES), desc="Training"):
        env.reset()  # Reset environment
        episode_reward = 0
        visited_pois = [False] * len(env.points_of_interest)
        current_poi_index = 0
        
        # Get initial state
        current_poi = env.points_of_interest[current_poi_index]
        obstacles = get_obstacles_from_grid(env)
        
        state = agent.get_state(
            (env.camera.x, env.camera.y),
            (current_poi.x, current_poi.y),
            obstacles,
            visited_pois
        )
        
        prev_distance = np.sqrt(
            (current_poi.x - env.camera.x)**2 + 
            (current_poi.y - env.camera.y)**2
        )
        
        for step in range(MAX_STEPS):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            collision = False
            if action == 0:  # Move forward
                if not env.camera.move_forward(env):
                    collision = True
            elif action == 1:  # Turn left
                env.camera.rotate(-10)
            elif action == 2:  # Turn right
                env.camera.rotate(10)
            # Action 3 is no movement
            
            # Calculate new distance to current POI
            current_distance = np.sqrt(
                (current_poi.x - env.camera.x)**2 + 
                (current_poi.y - env.camera.y)**2
            )
            
            # Check if reached POI (within 2 units)
            reached_poi = current_distance < 2.0
            
            # Calculate reward
            reward = calculate_reward(env, prev_distance, current_distance, reached_poi, collision)
            episode_reward += reward
            
            # Update POI status if reached
            if reached_poi and not visited_pois[current_poi_index]:
                visited_pois[current_poi_index] = True
                current_poi_index = (current_poi_index + 1) % len(env.points_of_interest)
                current_poi = env.points_of_interest[current_poi_index]
            
            # Get next state
            obstacles = get_obstacles_from_grid(env)
            next_state = agent.get_state(
                (env.camera.x, env.camera.y),
                (current_poi.x, current_poi.y),
                obstacles,
                visited_pois
            )
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, False)
            
            # Update model
            loss = agent.update_model()
            if loss is not None:
                writer.add_scalar('Loss/train', loss, agent.steps_done)
            
            # Update state and distance
            state = next_state
            prev_distance = current_distance
            
            # Check if all POIs visited
            if all(visited_pois):
                break
        
        # Log episode stats
        writer.add_scalar('Reward/episode', episode_reward, episode)
        writer.add_scalar('POIs/visited', sum(visited_pois), episode)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(save_dir / "best_model.pth")
        
        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_model(save_dir / f"checkpoint_episode_{episode+1}.pth")
    
    # Save final model
    agent.save_model(save_dir / "final_model.pth")
    writer.close()

if __name__ == "__main__":
    train()
