import numpy as np
import cv2
import math
import heapq
import json
import logging
from typing import List, Tuple, Optional
from enum import Enum

# Constants
GRID_SIZE = 50
CAMERA_FOV = 60
MOVEMENT_SPEED = 1.0
ROTATION_SPEED = 5.0

class MovementPattern:
    DIRECT = "direct"
    CIRCULAR = "circular"
    SMOOTH_APPROACH = "smooth"

class Wall:
    def __init__(self, x: int, y: int, type: str):
        self.x = x
        self.y = y
        self.type = type

class Camera:
    """Camera class for controlling movement and rotation"""
    def __init__(self, x=10, y=10, angle=0, fov=60):
        self.x = x
        self.y = y
        self.angle = angle
        self.fov = fov
        self.target_point = None
        self.path = []
        self.current_path_index = 0
        self.movement_pattern = MovementPattern.DIRECT
        
        # Store initial position for reset
        self.initial_x = x
        self.initial_y = y
        self.initial_angle = angle

    def rotate(self, angle_delta):
        """Rotate camera by given angle"""
        self.angle = (self.angle + angle_delta) % 360
        return True

    def move_forward(self, env, distance=1.0):
        """Move camera forward in current direction"""
        angle_rad = math.radians(self.angle)
        new_x = self.x + distance * math.cos(angle_rad)
        new_y = self.y + distance * math.sin(angle_rad)
        
        # Check if new position is valid
        if not env.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            return True
        return False

    def reset(self):
        """Reset camera to initial position"""
        self.x = self.initial_x
        self.y = self.initial_y
        self.angle = self.initial_angle
        self.path = []
        self.current_path_index = 0
        self.target_point = None
        self.movement_pattern = MovementPattern.DIRECT

    def set_target(self, target, path):
        """Set target point and path"""
        self.target_point = target
        self.path = path
        self.current_path_index = 0

    def move_along_path(self, env):
        """Move along the current path"""
        if not self.path or self.current_path_index >= len(self.path):
            return True

        target = self.path[self.current_path_index]
        dx = target[0] - self.x
        dy = target[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance < 0.1:  # Close enough to current waypoint
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                return True
            return False

        # Calculate angle to target
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (target_angle - self.angle) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # Rotate towards target
        if abs(angle_diff) > 5:
            self.rotate(math.copysign(min(5, abs(angle_diff)), angle_diff))
            return False

        # Move forward
        return not self.move_forward(env, min(1.0, distance))

    def get_view_lines(self):
        """Get camera view lines for rendering"""
        angle_rad = math.radians(self.angle)
        fov_rad = math.radians(self.fov)
        
        # Calculate view line endpoints
        left_angle = angle_rad - fov_rad/2
        right_angle = angle_rad + fov_rad/2
        
        view_length = 20
        left_x = self.x + view_length * math.cos(left_angle)
        left_y = self.y + view_length * math.sin(left_angle)
        right_x = self.x + view_length * math.cos(right_angle)
        right_y = self.y + view_length * math.sin(right_angle)
        
        # Return lines in format [[x1, x2], [y1, y2]] for both left and right lines
        left_line = [[self.x, left_x], [self.y, left_y]]
        right_line = [[self.x, right_x], [self.y, right_y]]
        
        return left_line, right_line

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
        """A* pathfinding with improved wall avoidance and alternative path finding"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # If goal is unreachable, try to find nearest accessible point
        if self.env.check_collision(*goal):
            logging.debug(f"Original goal {goal} is in collision, searching for alternative")
            best_alt = None
            min_dist = float('inf')
            search_radius = 5  # Increased search radius
            
            # Search in expanding squares around the goal
            for radius in range(1, search_radius + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:  # Only check the perimeter
                            alt_x, alt_y = goal[0] + dx, goal[1] + dy
                            if (0 <= alt_x < GRID_SIZE and 0 <= alt_y < GRID_SIZE and 
                                not self.env.check_collision(alt_x, alt_y)):
                                dist = abs(alt_x - goal[0]) + abs(alt_y - goal[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    best_alt = (alt_x, alt_y)
                
                if best_alt:  # If found a valid point, use it
                    logging.debug(f"Found alternative goal at {best_alt}")
                    goal = best_alt
                    break
            
            if not best_alt:
                logging.warning(f"No accessible point found near goal {goal}")
                return []

        # Standard A* implementation
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                dx = abs(next_pos[0] - current[0])
                dy = abs(next_pos[1] - current[1])
                movement_cost = 1.4 if dx + dy == 2 else 1.0
                
                new_cost = cost_so_far[current] + movement_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        if goal not in came_from:
            logging.warning(f"No path found to goal {goal}")
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
                new_x = smoothed[i][0] + 0.3 * (mid_x - smoothed[i][0])
                new_y = smoothed[i][1] + 0.3 * (mid_y - smoothed[i][1])
                
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
        self.initial_config = None
        self._walls = []  # Store walls
        
        # Load configuration
        if config_file:
            self.config = EnvironmentConfig.load_from_file(config_file)
            self.initial_config = config_file
        else:
            self.config = EnvironmentConfig.get_default_config()
        
        # Initialize camera with config
        start_pos = self.config["start_position"]
        self.camera = Camera(x=start_pos["x"], y=start_pos["y"], angle=start_pos["angle"], fov=CAMERA_FOV)
        
        # Setup environment
        self.setup_environment()
        
        # Initialize path planner
        self.path_planner = PathPlanner(self)

    @property
    def walls(self):
        """Get all walls in the environment"""
        return self._walls

    def setup_environment(self):
        """Setup environment based on configuration"""
        # Clear existing walls and POIs
        self._walls = []
        self.points_of_interest = []
        self.grid.fill(0)

        # Add walls from configuration
        for wall_config in self.config["walls"]:
            wall_type = wall_config["type"]
            x, y = wall_config["x"], wall_config["y"]
            
            if wall_type == "vertical":
                wall = Wall(x, y, "vertical")
            elif wall_type == "horizontal":
                wall = Wall(x, y, "horizontal")
            elif wall_type == "rectangle":
                wall = Wall(x, y, "rectangle")
            
            self._walls.append(wall)
            self.add_wall_to_grid(wall)

        # Add points of interest from configuration
        for poi_config in self.config["points_of_interest"]:
            poi = PointOfInterest(
                x=poi_config["x"],
                y=poi_config["y"],
                importance=poi_config["importance"]
            )
            self.points_of_interest.append(poi)

    def add_wall_to_grid(self, wall: Wall):
        """Add wall to grid"""
        # Get wall configuration from config
        wall_config = next(
            (w for w in self.config["walls"] if w["x"] == wall.x and w["y"] == wall.y),
            None
        )
        
        if wall_config is None:
            logging.warning(f"No configuration found for wall at ({wall.x}, {wall.y})")
            return
            
        if wall.type == "vertical":
            length = wall_config.get("length", 5)  # Default length of 5
            self.grid[wall.x:wall.x+length, wall.y] = 1
        elif wall.type == "horizontal":
            length = wall_config.get("length", 5)  # Default length of 5
            self.grid[wall.x, wall.y:wall.y+length] = 1
        elif wall.type == "rectangle":
            width = wall_config.get("width", 3)   # Default width of 3
            height = wall_config.get("height", 3)  # Default height of 3
            self.grid[wall.x:wall.x+width, wall.y:wall.y+height] = 1

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

    def reset(self):
        """Reset environment to initial state"""
        # Reset camera
        self.camera.reset()
        
        # Reset walls and POIs to initial configuration
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.points_of_interest = []
        
        # Reload initial configuration if it exists
        if self.initial_config:
            self.config = EnvironmentConfig.load_from_file(self.initial_config)
        else:
            # Default configuration if no file was provided
            self.config = EnvironmentConfig.get_default_config()
        
        # Re-setup environment
        self.setup_environment()

    def check_collision(self, x: float, y: float) -> bool:
        """Enhanced collision detection with safety margin"""
        cell_x = int(x)
        cell_y = int(y)
        
        # Check if position is out of bounds
        if not (0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE):
            return True
            
        # First check the exact cell
        if self.grid[cell_x, cell_y] == 1:
            return True
            
        # Then check immediate neighbors with a smaller margin
        margin = 0.5
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                check_x = int(x + i * margin)
                check_y = int(y + j * margin)
                
                if (0 <= check_x < GRID_SIZE and 
                    0 <= check_y < GRID_SIZE and 
                    self.grid[check_x, check_y] == 1):
                    
                    # Calculate exact distance to wall
                    wall_dist = math.sqrt(
                        (x - check_x) ** 2 + 
                        (y - check_y) ** 2
                    )
                    
                    if wall_dist < 0.3:  # Reduced from 0.5
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
        img_size = GRID_SIZE * 15
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Draw walls
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == 1:
                    cv2.rectangle(img,
                                (j * 15, i * 15),
                                ((j + 1) * 15 - 1, (i + 1) * 15 - 1),
                                (0, 0, 0), -1)
        
        # Draw points of interest
        for poi in self.points_of_interest:
            color = (0, int(255 * poi.importance), 0)
            cv2.circle(img,
                      (int(poi.x * 15 + 15/2),
                       int(poi.y * 15 + 15/2)),
                      int(15/2),
                      color, -1)
        
        # Draw path if it exists
        if self.camera.path:
            for i in range(len(self.camera.path) - 1):
                start = (int(self.camera.path[i][1] * 15), 
                        int(self.camera.path[i][0] * 15))
                end = (int(self.camera.path[i+1][1] * 15), 
                      int(self.camera.path[i+1][0] * 15))
                cv2.line(img, start, end, (0, 0, 255), 1)
        
        # Draw camera
        camera_pos = (int(self.camera.x * 15), int(self.camera.y * 15))
        cv2.circle(img, camera_pos, int(15/2), (255, 0, 0), -1)
        
        # Draw camera direction and FOV
        left_line, right_line = self.camera.get_view_lines()
        
        # Convert view lines to pixel coordinates
        left_start = (int(left_line[0][0] * 15), int(left_line[1][0] * 15))
        left_end = (int(left_line[0][1] * 15), int(left_line[1][1] * 15))
        right_start = (int(right_line[0][0] * 15), int(right_line[1][0] * 15))
        right_end = (int(right_line[0][1] * 15), int(right_line[1][1] * 15))
        
        # Draw FOV lines
        cv2.line(img, left_start, left_end, (255, 0, 0), 1)
        cv2.line(img, right_start, right_end, (255, 0, 0), 1)
        
        return img

class PointOfInterest:
    def __init__(self, x: int, y: int, importance: float):
        self.x = x
        self.y = y
        self.importance = importance

def main():
    # You can specify a config file or use default configuration
    env = Environment("custom_environment.json" if len(sys.argv) > 1 else None)
    auto_mode = False
    current_poi_index = 0
    visited_pois = set()  # Keep track of visited POIs
    movement_patterns = [MovementPattern.DIRECT, MovementPattern.CIRCULAR, MovementPattern.SMOOTH_APPROACH]
    current_pattern = 0
    
    print("Controls:")
    print("W/S: Move forward/backward")
    print("A/D: Rotate left/right")
    print("P: Toggle automatic path following")
    print("M: Change movement pattern")
    print("O: Save current configuration")
    print("R: Reset environment")
    print("ESC: Exit")
    
    while True:
        img = env.render()
        cv2.imshow("Camera Path Planning", img)
        
        if auto_mode and env.camera.path:
            if env.camera.move_along_path(env):
                visited_pois.add(current_poi_index)
                logging.debug(f"Reached POI at index: {current_poi_index}")
                
                # Find next unvisited POI
                original_index = current_poi_index
                while True:
                    current_poi_index = (current_poi_index + 1) % len(env.points_of_interest)
                    if current_poi_index not in visited_pois or current_poi_index == original_index:
                        break
                
                if len(visited_pois) == len(env.points_of_interest):
                    logging.info("All POIs have been visited!")
                    visited_pois.clear()  # Reset for next round
                
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
                    # Skip to next POI if current is unreachable
                    current_poi_index = (current_poi_index + 1) % len(env.points_of_interest)
        
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
            env.camera.rotate(-ROTATION_SPEED)
        elif key == ord('d'):
            env.camera.rotate(ROTATION_SPEED)
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
        elif key == ord('r'):
            env.reset()
            visited_pois.clear()
            current_poi_index = 0
            logging.info("Environment reset")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
