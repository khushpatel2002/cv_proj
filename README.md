# Camera Navigation with Reinforcement Learning

A computer vision project that implements an intelligent camera navigation system using Deep Q-Learning. The system learns to autonomously navigate through a 2D environment, visiting points of interest while avoiding obstacles.

## Features

- **Intelligent Camera System**
  - Autonomous navigation using Deep Q-Learning
  - Field of view (FOV) visualization
  - Collision avoidance
  - Multiple movement patterns (Direct, Circular, Smooth Approach)

- **Environment System**
  - Customizable 2D grid-based environment
  - Support for various wall types (vertical, horizontal, rectangle)
  - Points of Interest (POI) with importance weights
  - JSON-based environment configuration

- **Reinforcement Learning**
  - Deep Q-Network (DQN) implementation
  - Experience replay for stable learning
  - Prioritized movement actions
  - TensorBoard integration for training visualization

## Requirements

```
numpy
torch
opencv-python
tensorboard
tqdm
```

## Project Structure

- `dev.py` - Core environment and camera implementation
- `rl_agent.py` - DQN agent implementation
- `train_rl.py` - Training script for the RL agent
- `evaluate.py` - Evaluation script for trained models

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd cv_proj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new model:
```bash
python train_rl.py
```

Training parameters can be modified in `train_rl.py`:
- `EPISODES`: Number of training episodes
- `MAX_STEPS`: Maximum steps per episode
- `LEARNING_RATE`: Learning rate for the optimizer
- `BATCH_SIZE`: Batch size for training
- `EPSILON_START`: Initial exploration rate
- `EPSILON_END`: Final exploration rate

### Evaluation

To evaluate a trained model:
```bash
python evaluate.py
```

### Environment Configuration

Create custom environments by modifying `custom_environment.json`:
```json
{
    "walls": [
        {"type": "vertical", "x": 15, "y": 15, "length": 10},
        {"type": "horizontal", "x": 15, "y": 15, "length": 10}
    ],
    "points_of_interest": [
        {"x": 10, "y": 10, "importance": 1.0}
    ],
    "start_position": {"x": 20, "y": 20, "angle": 0}
}
```
