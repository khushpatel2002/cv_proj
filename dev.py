import numpy as np
import cv2
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
GRID_SIZE = 40
CELL_SIZE = 15
CAMERA_FOV = 60
SMOOTHING_FACTOR = 0.3
MOVEMENT_SPEED = 0.3
ROTATION_SPEED = 3

class MovementPattern:
    DIRECT = "direct"
    CIRCULAR = "circular"
    SMOOTH_APPROACH = "smooth"

class EnvironmentConfig:
    @staticmethod
    def load_from_file(filename: str) -> dict:
        """Load environment configuration from a JSON file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            return EnvironmentConfig.get_default_config()

    @staticmethod
    def get_default_config() -> dict:
        """Get default environment configuration"""
        return {
            "walls": [
                {"type": "vertical", "x": 15, "y": 15, "length": 10},
                {"type": "horizontal", "x": 15, "y": 15, "length": 10},
                {"type": "rectangle", "x": 25, "y": 25, "width": 5, "height": 5}
            ],
            "points_of_interest": [
                {"x": 10, "y": 10, "importance": 1.0},
                {"x": 30, "y": 30, "importance": 0.9},
                {"x": 10, "y": 30, "importance": 0.8},
                {"x": 30, "y": 10, "importance": 0.7},
                {"x": 20, "y": 35, "importance": 0.6}
            ],
            "start_position": {"x": 20, "y": 20, "angle": 0}
        }

    @staticmethod
    def save_to_file(config: dict, filename: str):
        """Save environment configuration to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

@dataclass
class Camera:
    x: float
    y: float
    angle: float
    fov: float
    target_point: Optional[Tuple[float, float]] = None
    path: List[Tuple[float, float]] = None
    current_path_index: int = 0
    movement_pattern: str = MovementPattern.DIRECT
    
    def get_view_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the lines representing the camera's field of view"""
        angle_rad = math.radians(self.angle)
        fov_rad = math.radians(self.fov)
        
        left_angle = angle_rad - fov_rad/2
        right_angle = angle_rad + fov_rad/2
        
        view_distance = GRID_SIZE
        
        left_x = self.x + view_distance * math.cos(left_angle)
        left_y = self.y + view_distance * math.sin(left_angle)
        right_x = self.x + view_distance * math.cos(right_angle)
        right_y = self.y + view_distance * math.sin(right_angle)
        
        return (np.array([self.x, left_x]), np.array([self.y, left_y])), \
               (np.array([self.x, right_x]), np.array([self.y, right_y]))

    def set_target(self, target: Tuple[float, float], path: List[Tuple[float, float]]):
        self.target_point = target
        self.path = path
        self.current_path_index = 0

    def move_along_path(self, env) -> bool:
        """Move camera along the current path with improved collision handling"""
        if not self.path or self.current_path_index >= len(self.path):
            return True

        # Get next point in path
        target = self.path[self.current_path_index]
        
        # Calculate direction to target
        dx = target[0] - self.x
        dy = target[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:  # Reached current waypoint
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                return True
            return False
        
        # Calculate desired angle based on movement pattern
        if self.movement_pattern == MovementPattern.DIRECT:
            desired_angle = math.degrees(math.atan2(dy, dx))
        elif self.movement_pattern == MovementPattern.CIRCULAR:
            circle_radius = 1.0  
            progress = self.current_path_index / len(self.path)
            circular_offset = circle_radius * math.sin(progress * 2 * math.pi)
            desired_angle = math.degrees(math.atan2(dy + circular_offset, dx))
        else:  # SMOOTH_APPROACH
            progress = self.current_path_index / len(self.path)
            s_curve_offset = 0.5 * math.sin(progress * 2 * math.pi)  
            desired_angle = math.degrees(math.atan2(dy + s_curve_offset, dx))
        
        # Smoothly rotate towards desired angle
        angle_diff = (desired_angle - self.angle + 180) % 360 - 180
        if abs(angle_diff) > 1:  
            self.angle += min(max(angle_diff, -ROTATION_SPEED), ROTATION_SPEED)
            return False
        
        # Move towards target with collision check
        move_distance = min(MOVEMENT_SPEED, distance)
        move_dx = move_distance * math.cos(math.radians(self.angle))
        move_dy = move_distance * math.sin(math.radians(self.angle))
        
        # Try to move, if collision detected, try to find alternative path
        if env.move_camera(move_dx, move_dy):
            # If stuck, request new path
            self.path = env.path_planner.find_path((self.x, self.y), target)
            if not self.path:
                return True  # Give up if no path found
            self.current_path_index = 0
            return False
        
        return False

class PathPlanner:
    def __init__(self, env):
        self.env = env
        self.directions = [
            (0, 1),   # North
            (1, 1),   # Northeast
            (1, 0),   # East
            (1, -1),  # Southeast
            (0, -1),  # South
            (-1, -1), # Southwest
            (-1, 0),  # West
            (-1, 1)   # Northwest
        ]

    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        neighbors = []
        x, y = pos
        
        for dx, dy in self.directions:
            new_x, new_y = x + dx, y + dy
            
            # Check if position is within grid bounds
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                # Check for collision with expanded margin around walls
                collision = False
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        check_x = int(new_x + i)
                        check_y = int(new_y + j)
                        if (0 <= check_x < GRID_SIZE and 
                            0 <= check_y < GRID_SIZE and 
                            self.env.grid[check_x, check_y] == 1):
                            collision = True
                            break
                    if collision:
                        break
                
                if not collision:
                    neighbors.append((new_x, new_y))
        
        return neighbors

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* pathfinding with improved wall avoidance"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        if self.env.check_collision(*goal):
            # If goal is in wall, find nearest free space
            min_dist = float('inf')
            new_goal = None
            for x in range(max(0, goal[0]-3), min(GRID_SIZE, goal[0]+4)):
                for y in range(max(0, goal[1]-3), min(GRID_SIZE, goal[1]+4)):
                    if not self.env.check_collision(x, y):
                        dist = abs(x - goal[0]) + abs(y - goal[1])
                        if dist < min_dist:
                            min_dist = dist
                            new_goal = (x, y)
            if new_goal:
                goal = new_goal
            else:
                return []

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                # Calculate movement cost (diagonal movement costs more)
                dx = abs(next_pos[0] - current[0])
                dy = abs(next_pos[1] - current[1])
                movement_cost = 1.4 if dx + dy == 2 else 1.0
                
                new_cost = cost_so_far[current] + movement_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:
            return []
            
        path = []
        current = goal
        while current is not None:
            path.append((float(current[0]), float(current[1])))
            current = came_from[current]
        path.reverse()
        
        return self.smooth_path(path)

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth the path while maintaining safe distance from walls"""
        if len(path) <= 2:
            return path
            
        smoothed = list(path)
        iterations = 3  
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                old_pos = smoothed[i]
                
                # Calculate the midpoint between previous and next points
                mid_x = (smoothed[i-1][0] + smoothed[i+1][0]) / 2
                mid_y = (smoothed[i-1][1] + smoothed[i+1][1]) / 2
                
                # Move current point towards midpoint with reduced smoothing
                new_x = smoothed[i][0] + SMOOTHING_FACTOR * (mid_x - smoothed[i][0])
                new_y = smoothed[i][1] + SMOOTHING_FACTOR * (mid_y - smoothed[i][1])
                
                # Check if new position is safe (with margin)
                if not self.env.check_collision(new_x, new_y):
                    collision = False
                    for dx in [-0.5, 0, 0.5]:
                        for dy in [-0.5, 0, 0.5]:
                            if self.env.check_collision(new_x + dx, new_y + dy):
                                collision = True
                                break
                        if collision:
                            break
                    
                    if not collision:
                        smoothed[i] = (new_x, new_y)
                    
        return smoothed

class Environment:
    def __init__(self, config_file: str = None):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.points_of_interest = []
        
        # Load configuration
        if config_file:
            self.config = EnvironmentConfig.load_from_file(config_file)
        else:
            self.config = EnvironmentConfig.get_default_config()
        
        # Initialize camera with config
        start_pos = self.config["start_position"]
        self.camera = Camera(start_pos["x"], start_pos["y"], start_pos["angle"], CAMERA_FOV)
        
        self.path_planner = PathPlanner(self)
        self.setup_environment()

    def setup_environment(self):
        """Setup environment based on configuration"""
        # Create outer walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Create walls from configuration
        for wall in self.config["walls"]:
            if wall["type"] == "vertical":
                self.grid[wall["x"]:wall["x"]+wall["length"], wall["y"]] = 1
            elif wall["type"] == "horizontal":
                self.grid[wall["x"], wall["y"]:wall["y"]+wall["length"]] = 1
            elif wall["type"] == "rectangle":
                self.grid[wall["x"]:wall["x"]+wall["width"], 
                         wall["y"]:wall["y"]+wall["height"]] = 1

        # Setup points of interest
        self.points_of_interest = [
            PointOfInterest(poi["x"], poi["y"], poi["importance"])
            for poi in self.config["points_of_interest"]
        ]
        # Sort by importance
        self.points_of_interest.sort(key=lambda x: x.importance, reverse=True)

    def save_current_config(self, filename: str):
        """Save current environment configuration"""
        config = {
            "walls": self.config["walls"],
            "points_of_interest": [
                {"x": poi.x, "y": poi.y, "importance": poi.importance}
                for poi in self.points_of_interest
            ],
            "start_position": {
                "x": self.camera.x,
                "y": self.camera.y,
                "angle": self.camera.angle
            }
        }
        EnvironmentConfig.save_to_file(config, filename)

    def check_collision(self, x: float, y: float) -> bool:
        """Enhanced collision detection with safety margin"""
        cell_x = int(x)
        cell_y = int(y)
        
        # Check surrounding cells for walls
        margin = 1
        for i in range(-margin, margin + 1):
            for j in range(-margin, margin + 1):
                check_x = cell_x + i
                check_y = cell_y + j
                
                if (0 <= check_x < GRID_SIZE and 
                    0 <= check_y < GRID_SIZE and 
                    self.grid[check_x, check_y] == 1):
                    # Calculate distance to wall
                    wall_dist = math.sqrt((x - check_x)**2 + (y - check_y)**2)
                    if wall_dist < 0.5:  
                        return True
        
        return False

    def move_camera(self, dx: float, dy: float) -> bool:
        """Move camera with enhanced collision checking"""
        new_x = self.camera.x + dx
        new_y = self.camera.y + dy
        
        # Check if new position would be in collision
        if self.check_collision(new_x, new_y):
            # Try to slide along walls
            if not self.check_collision(self.camera.x + dx, self.camera.y):
                self.camera.x += dx
            elif not self.check_collision(self.camera.x, self.camera.y + dy):
                self.camera.y += dy
            return True
        
        self.camera.x = new_x
        self.camera.y = new_y
        return False

    def render(self) -> np.ndarray:
        # Create RGB image
        img_size = GRID_SIZE * CELL_SIZE
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Draw walls
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == 1:
                    cv2.rectangle(img,
                                (j * CELL_SIZE, i * CELL_SIZE),
                                ((j + 1) * CELL_SIZE - 1, (i + 1) * CELL_SIZE - 1),
                                (0, 0, 0), -1)
        
        # Draw points of interest
        for poi in self.points_of_interest:
            color = (0, int(255 * poi.importance), 0)
            cv2.circle(img,
                      (int(poi.x * CELL_SIZE + CELL_SIZE/2),
                       int(poi.y * CELL_SIZE + CELL_SIZE/2)),
                      int(CELL_SIZE/2),
                      color, -1)
        
        # Draw path if it exists
        if self.camera.path:
            for i in range(len(self.camera.path) - 1):
                start = (int(self.camera.path[i][1] * CELL_SIZE), 
                        int(self.camera.path[i][0] * CELL_SIZE))
                end = (int(self.camera.path[i+1][1] * CELL_SIZE), 
                      int(self.camera.path[i+1][0] * CELL_SIZE))
                cv2.line(img, start, end, (0, 0, 255), 1)
        
        # Draw camera
        camera_pos = (int(self.camera.x * CELL_SIZE), int(self.camera.y * CELL_SIZE))
        cv2.circle(img, camera_pos, int(CELL_SIZE/2), (255, 0, 0), -1)
        
        # Draw camera direction and FOV
        left_line, right_line = self.camera.get_view_lines()
        
        # Convert view lines to pixel coordinates
        left_start = (int(left_line[0][0] * CELL_SIZE), int(left_line[1][0] * CELL_SIZE))
        left_end = (int(left_line[0][1] * CELL_SIZE), int(left_line[1][1] * CELL_SIZE))
        right_start = (int(right_line[0][0] * CELL_SIZE), int(right_line[1][0] * CELL_SIZE))
        right_end = (int(right_line[0][1] * CELL_SIZE), int(right_line[1][1] * CELL_SIZE))
        
        # Draw FOV lines
        cv2.line(img, left_start, left_end, (255, 0, 0), 1)
        cv2.line(img, right_start, right_end, (255, 0, 0), 1)
        
        return img

@dataclass
class PointOfInterest:
    x: int
    y: int
    importance: float  

def main():
    # You can specify a config file or use default configuration
    env = Environment("custom_environment.json" if len(sys.argv) > 1 else None)
    auto_mode = False
    current_poi_index = 0
    movement_patterns = [MovementPattern.DIRECT, MovementPattern.CIRCULAR, MovementPattern.SMOOTH_APPROACH]
    current_pattern = 0
    
    print("Controls:")
    print("W/S: Move forward/backward")
    print("A/D: Rotate left/right")
    print("P: Toggle automatic path following")
    print("M: Change movement pattern")
    print("O: Save current configuration")
    print("ESC: Exit")
    
    while True:
        img = env.render()
        cv2.imshow("Camera Path Planning", img)
        
        if auto_mode and env.camera.path:
            if env.camera.move_along_path(env):
                logging.debug(f"Reached POI at index: {current_poi_index}")
                current_poi_index = (current_poi_index + 1) % len(env.points_of_interest)
                next_poi = env.points_of_interest[current_poi_index]
                logging.debug(f"Next POI: {next_poi}")
                path = env.path_planner.find_path((env.camera.x, env.camera.y), 
                                                (next_poi.x, next_poi.y))
                if path:
                    logging.debug(f"Path found to POI: {path}")
                    env.camera.movement_pattern = movement_patterns[current_pattern]
                    env.camera.set_target((next_poi.x, next_poi.y), path)
                else:
                    logging.warning(f"No path found to POI at index: {current_poi_index}")
        
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break
        elif key == ord('w'):
            dx = MOVEMENT_SPEED * math.cos(math.radians(env.camera.angle))
            dy = MOVEMENT_SPEED * math.sin(math.radians(env.camera.angle))
            env.move_camera(dx, dy)
        elif key == ord('s'):
            dx = -MOVEMENT_SPEED * math.cos(math.radians(env.camera.angle))
            dy = -MOVEMENT_SPEED * math.sin(math.radians(env.camera.angle))
            env.move_camera(dx, dy)
        elif key == ord('a'):
            env.camera.angle += ROTATION_SPEED
        elif key == ord('d'):
            env.camera.angle -= ROTATION_SPEED
        elif key == ord('p'):
            auto_mode = not auto_mode
            logging.info(f"Automatic mode {'enabled' if auto_mode else 'disabled'}.")
            if auto_mode:
                next_poi = env.points_of_interest[current_poi_index]
                logging.debug(f"Starting auto mode with POI: {next_poi}")
                path = env.path_planner.find_path((env.camera.x, env.camera.y), 
                                                (next_poi.x, next_poi.y))
                if path:
                    logging.debug(f"Initial path found to POI: {path}")
                    env.camera.movement_pattern = movement_patterns[current_pattern]
                    env.camera.set_target((next_poi.x, next_poi.y), path)
                else:
                    logging.warning(f"No initial path found to POI at index: {current_poi_index}")
        elif key == ord('m'):
            current_pattern = (current_pattern + 1) % len(movement_patterns)
            if env.camera.path:
                env.camera.movement_pattern = movement_patterns[current_pattern]
        elif key == ord('o'):
            env.save_current_config("custom_environment.json")
            print("Configuration saved to custom_environment.json")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
