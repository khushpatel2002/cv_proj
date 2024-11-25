import numpy as np
import torch
import cv2
import logging
from pathlib import Path
from dev import Environment
from rl_agent import RLAgent
from train_rl import NUM_ACTIONS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model_path: str, render: bool = True):
    """Evaluate a trained model"""
    # Initialize environment and agent
    env = Environment("custom_environment.json")
    
    # Calculate state size
    state_size = 2 + 2 + 2 + (5 * 2) + len(env.points_of_interest)
    logger.info(f"State size: {state_size}")
    
    # Initialize agent and load model
    agent = RLAgent(state_size=state_size, action_size=NUM_ACTIONS)
    agent.load_model(model_path)
    agent.epsilon = 0.5  # Increased exploration rate
    
    # Initialize tracking variables
    visited_pois = [False] * len(env.points_of_interest)
    current_poi_index = 0
    total_reward = 0
    steps = 0
    last_distance = None
    
    # Get initial state
    current_poi = env.points_of_interest[current_poi_index]
    logger.info(f"Initial POI position: ({current_poi.x}, {current_poi.y})")
    logger.info(f"Initial camera position: ({env.camera.x}, {env.camera.y})")
    
    # Get obstacles from grid
    obstacles = []
    for i in range(env.grid.shape[0]):
        for j in range(env.grid.shape[1]):
            if env.grid[i,j] == 1:  # If there's a wall
                obstacles.append((i, j))
    
    state = agent.get_state(
        (env.camera.x, env.camera.y),
        (current_poi.x, current_poi.y),
        obstacles,
        visited_pois
    )
    
    logger.info(f"Initial state shape: {state.shape}")
    
    while not all(visited_pois) and steps < 1000:
        # Select action
        action = agent.select_action(state)
        logger.debug(f"Step {steps}: Selected action {action}")
        
        # Execute action
        if action == 0:  # Move forward
            collision = not env.camera.move_forward(env, distance=0.5)  # Reduced movement distance
            logger.debug(f"Moving forward, collision: {collision}")
        elif action == 1:  # Turn left
            env.camera.rotate(-5)  # Reduced rotation angle
            collision = False
            logger.debug("Turning left")
        elif action == 2:  # Turn right
            env.camera.rotate(5)  # Reduced rotation angle
            collision = False
            logger.debug("Turning right")
        else:  # No movement
            collision = False
            logger.debug("No movement")
        
        # Calculate distance to current POI
        current_distance = np.sqrt(
            (current_poi.x - env.camera.x)**2 + 
            (current_poi.y - env.camera.y)**2
        )
        logger.debug(f"Distance to POI: {current_distance:.2f}")
        
        # Calculate reward
        reward = 0
        if collision:
            reward = -0.5  # Reduced collision penalty
        elif last_distance is not None:
            # Reward for getting closer to POI
            if current_distance < last_distance:
                reward += 0.5  # Increased reward for progress
            else:
                reward -= 0.1  # Reduced penalty for moving away
            
            # Add small penalty for not moving
            if action == 3:  # No movement action
                reward -= 0.2  # Penalize standing still
        
        last_distance = current_distance
        total_reward += reward
        
        # Check if reached POI
        reached_poi = current_distance < 1.5  # Increased reach radius
        
        # Update POI status if reached
        if reached_poi and not visited_pois[current_poi_index]:
            visited_pois[current_poi_index] = True
            logger.info(f"Reached POI {current_poi_index}")
            current_poi_index = (current_poi_index + 1) % len(env.points_of_interest)
            current_poi = env.points_of_interest[current_poi_index]
        
        # Get next state
        obstacles = []
        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                if env.grid[i,j] == 1:  # If there's a wall
                    obstacles.append((i, j))
                    
        next_state = agent.get_state(
            (env.camera.x, env.camera.y),
            (current_poi.x, current_poi.y),
            obstacles,
            visited_pois
        )
        
        # Update state
        state = next_state
        steps += 1
        
        # Render if requested
        if render:
            img = env.render()
            cv2.imshow("Camera Path Planning (RL)", img)
            key = cv2.waitKey(50)
            if key == 27:  # ESC
                break
    
    # Print evaluation results
    logger.info(f"Evaluation completed in {steps} steps")
    logger.info(f"POIs visited: {sum(visited_pois)}/{len(visited_pois)}")
    logger.info(f"Total reward: {total_reward}")
    
    if render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use the best model for evaluation
    model_path = Path("/Users/khushpatel2002/cv_proj/models/final_model.pth")
    if not model_path.exists():
        logger.error(f"No model found at {model_path}")
    else:
        evaluate(str(model_path))
