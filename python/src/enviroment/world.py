from typing import List, Tuple, Optional
import numpy as np
from enviroment.obstacles import Obstacle
from utils.debug_helper import DebugHelper

class World:
    """
    Represents the robot's operational environment with obstacles and coordinate systems.
    
    This class manages the world representation including:
    - World boundaries and dimensions
    - Obstacle placement and inflation
    - Coordinate system transformations
    - Robot origin reference
    
    Attributes:
        width (int): World width in display coordinates
        height (int): World height in display coordinates
        robot_origin (Tuple[int, int]): Robot reference point in display coordinates
        obstacles (List[Obstacle]): List of environment obstacles
        inflated_obstacles (List[Obstacle]): Obstacles with safety margins
        inflation (float): Safety margin size for obstacle inflation
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 robot_origin: Tuple[int, int], 
                 obstacles: Optional[List[Obstacle]] = None) -> None:
        """
        Initializes world environment with given dimensions and obstacles.
        
        Args:
            width: World width in display coordinates
            height: World height in display coordinates
            robot_origin: Reference point for robot coordinates
            obstacles: List of obstacles in environment (empty if None)
        """
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.obstacles = obstacles if obstacles is not None else []
        self.debug_helper = DebugHelper(debug=False)
        
        # Safety margin parameters
        self.inflation = 5.0  # Obstacle inflation size for safety
        self.inflated_obstacles = self._create_inflated_obstacles()
        
    def _create_inflated_obstacles(self) -> List[Obstacle]:
        """
        Creates inflated versions of all obstacles for safety margins.
        
        Inflates each obstacle by the safety margin to ensure proper
        clearance during motion planning.
        
        Returns:
            List[Obstacle]: Inflated versions of all obstacles
        """
        return [obstacle.inflate(self.inflation) for obstacle in self.obstacles]
        
    def convert_to_display(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """
        Converts robot coordinates to display coordinates.
        
        Transforms from robot coordinate frame (right-handed, origin at robot base)
        to display coordinate frame (top-left origin, y-down).
        
        Args:
            point: (x, y) coordinates in robot frame
            
        Returns:
            Tuple[int, int]: (x, y) coordinates in display frame
        
        Note:
            - Robot frame: Right-handed, origin at robot base
            - Display frame: Top-left origin, y-down
            - Y-axis is inverted between frames
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin
        
        # Transform and round to integer display coordinates
        return int(offset_x + robot_x), int(offset_y - robot_y)