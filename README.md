# Intelligent Camera Navigation System

A computer vision project that implements an autonomous camera navigation system with two different approaches:
1. A* Pathfinding (main branch)
2. Reinforcement Learning (rl_solution branch)

The system can navigate through a 2D environment, visiting points of interest while avoiding obstacles, with support for multiple movement patterns.

## Features

### Core Features (Both Implementations)
- **Environment System**
  - Customizable 2D grid-based environment
  - Support for various wall types:
    - Vertical walls
    - Horizontal walls
    - Rectangle obstacles
  - Points of Interest (POI) with importance weights
  - JSON-based environment configuration
  - Real-time visualization using OpenCV

### A* Implementation Features (main branch)
- **Intelligent Camera System**
  - Autonomous navigation with A* pathfinding
  - Field of view (FOV) visualization
  - Advanced collision avoidance
  - Multiple movement patterns:
    - Direct: Straight-line movement
    - Circular: Sinusoidal path movement
    - Smooth Approach: Gentle oscillating movement

### RL Implementation Features (rl_solution branch)
- **Deep Q-Learning Navigation**
  - Autonomous learning through environment interaction
  - Experience replay for stable training
  - Epsilon-greedy exploration strategy
  - Reward shaping based on:
    - Distance to points of interest
    - Collision avoidance
    - Field of view coverage
  - TensorBoard integration for training visualization

## Requirements

### Main Branch (A* Implementation)
numpy
opencv-python

### RL Branch
numpy
opencv-python
torch
tensorboard
tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khushpatel2002/cv_proj.git
cd cv_proj
```

2. Switch to desired implementation:
```bash
# For A* implementation
git checkout main

# For RL implementation
git checkout rl_solution
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### A* Implementation (main branch)

Run the main simulation:
```bash
python main.py
```

To use a custom environment configuration:
```bash
python main.py custom_environment.json
```

#### Controls (A* Implementation)
- `W/S`: Move camera forward/backward
- `A/D`: Rotate camera left/right
- `P`: Toggle automatic path following
- `M`: Change movement pattern
- `O`: Save current configuration
- `ESC`: Exit

### RL Implementation (rl_solution branch)

#### Training
```bash
python train_rl.py --episodes 1000 --batch-size 64
```

Training parameters:
- `--episodes`: Number of training episodes
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--epsilon-start`: Initial exploration rate
- `--epsilon-end`: Final exploration rate
- `--memory-size`: Size of replay memory

#### Evaluation
```bash
python evaluate.py --model-path models/dqn_model.pth
```

### Environment Configuration

Create custom environments by modifying `custom_environment.json`:
```json
{
    "walls": [
        {"type": "vertical", "x": 15, "y": 15, "length": 10},
        {"type": "horizontal", "x": 15, "y": 15, "length": 10},
        {"type": "rectangle", "x": 25, "y": 25, "width": 5, "height": 5}
    ],
    "points_of_interest": [
        {"x": 10, "y": 10, "importance": 1.0},
        {"x": 30, "y": 30, "importance": 0.9}
    ],
    "start_position": {"x": 20, "y": 20, "angle": 0}
}
```

## Implementation Details

### A* Implementation (main branch)

#### Key Components

1. **Camera Class**
   - Manages camera position, orientation, and movement
   - Implements FOV visualization
   - Handles different movement patterns

2. **Environment Class**
   - Manages the grid-based environment
   - Handles wall and obstacle placement
   - Manages points of interest
   - Provides collision detection

3. **PathPlanner Class**
   - Implements A* pathfinding algorithm
   - Provides path smoothing
   - Handles collision avoidance
   - Finds alternative paths when needed

#### Movement Patterns

1. **Direct Movement**
   - Straight-line path to target
   - Efficient for unobstructed paths

2. **Circular Movement**
   - Sinusoidal path following
   - Creates circular motion around the direct path
   - Useful for dynamic environment observation

3. **Smooth Approach**
   - Gentle oscillating movement
   - Provides smooth transitions
   - Ideal for careful navigation

### RL Implementation (rl_solution branch)

#### Key Components

1. **DQN Agent**
   - Deep Q-Network architecture
   - Experience replay memory
   - Target network for stable learning
   - Epsilon-greedy action selection

2. **State Representation**
   - Camera position and orientation
   - Distance and angle to points of interest
   - Local obstacle information
   - Current field of view coverage

3. **Action Space**
   - Move forward
   - Rotate left/right
   - Combined movement actions

4. **Reward Function**
   - Positive reward for visiting POIs
   - Negative reward for collisions
   - Shaped reward for approaching POIs
   - Coverage reward for maintaining good FOV

## Performance Comparison

### A* Implementation
- Deterministic and predictable behavior
- Optimal path finding in static environments
- Multiple movement patterns for different scenarios
- No training required

### RL Implementation
- Learns adaptive behaviors through experience
- Can handle dynamic environments
- Potentially more natural movement patterns
- Requires training but can generalize to new scenarios



## Branch Information

- `main`: A* pathfinding implementation
- `rl_solution`: Reinforcement learning implementation
