from typing import List, Tuple, Optional
from enviroment.robot import Robot
from enviroment.world import World
from utils.state import State
from utils.debug_helper import DebugHelper


class Planner:
    """
    Abstract base class for robot motion planners.
    
    Provides the interface that all planner implementations must follow.
    Handles basic initialization and defines the planning interface.
    
    Attributes:
        robot (Robot): Robot model containing kinematics and physical parameters
        world (World): World model containing obstacles and boundaries
    """

    def __init__(self, robot: Robot, world: World):
        """
        Initialize the base planner with robot and world models.

        Args:
            robot (Robot): Robot model containing kinematics and physical parameters
            world (World): World model containing obstacles and boundaries
        """
        self.robot = robot
        self.world = world

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             robot: Robot, final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """
        Plan a path from start position to goal position with specified final velocities.
        
        This is an abstract method that must be implemented by derived planner classes.
        
        Args:
            start (Tuple[float, float]): Starting position in workspace coordinates (x, y)
            goal (Tuple[float, float]): Goal position in workspace coordinates (x, y)
            robot (Robot): Robot model instance for kinematics calculations
            final_velocities (Tuple[float, float], optional): Desired final joint velocities 
                (omega_0, omega_1). Defaults to (0.0, 0.0) for zero final velocity.
            
        Returns:
            List[State]: Sequence of states representing the planned path from start to goal
            
        Raises:
            NotImplementedError: When called directly on base class
            ValueError: If no valid path is found
            PlanningTimeoutError: If planning exceeds time limit
        """
        raise NotImplementedError("Planner subclasses must implement plan() method")