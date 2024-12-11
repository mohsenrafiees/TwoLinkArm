from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

class RobotConstants:
    """
    Manages physical and operational constants for a 2-DOF robot arm.
    
    This class maintains all constant parameters needed for:
    - Robot geometry and physical properties
    - Joint limits and motion constraints
    - Planning and control parameters
    - Environmental considerations
    
    Attributes:
        BASE_RADIUS (float): Radius of robot base (meters)
        JOINT_LIMITS (Tuple[float, float]): Min/max joint angles (radians)
        MAX_VELOCITY (float): Maximum joint velocity (rad/s)
        MIN_VELOCITY (float): Minimum joint velocity (rad/s)
        MAX_ACCELERATION (float): Maximum joint acceleration (rad/s²)
        MAX_JERK (float): Maximum joint jerk (rad/s³)
        VELOCITY_RESOLUTION (float): Velocity discretization step (rad/s)
        DT (float): Time step for numerical integration (seconds)
        LINK_1 (float): Length of first link (meters)
        LINK_2 (float): Length of second link (meters)
        ROBOT_ORIGIN (Tuple[float, float]): Robot base position (meters)
        THETA_0_RESOLUTION (float): Angular resolution for joint 0 planning (rad)
        THETA_1_RESOLUTION (float): Angular resolution for joint 1 planning (rad)
        CONSIDER_GRAVITY (bool): Whether to include gravity in dynamics
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Initializes robot constants from configuration dictionary.
        
        Args:
            config: Dictionary containing robot parameters with the following keys:
                - base_radius (optional): Base radius in meters
                - joint_limits: [min, max] joint angles in radians
                - max_velocity: Maximum joint velocity in rad/s
                - min_velocity: Minimum joint velocity in rad/s
                - max_acceleration: Maximum joint acceleration in rad/s²
                - max_jerk: Maximum joint jerk in rad/s³
                - velocity_resolution: Velocity discretization in rad/s
                - dt: Time step in seconds
                - link_lengths: [link1, link2] lengths in meters
                - robot_origin (optional): (x, y) base position in meters
                - theta_0_resolution (optional): Joint 0 planning resolution in rad
                - theta_1_resolution (optional): Joint 1 planning resolution in rad
                - consider_gravity (optional): Boolean for gravity consideration
                
        Raises:
            KeyError: If required configuration parameters are missing
        """
        # Physical parameters
        self.BASE_RADIUS = config.get('base_radius', 10.0)
        self.JOINT_LIMITS = tuple(config['joint_limits'])
        self.LINK_1 = config['link_lengths'][0]
        self.LINK_2 = config['link_lengths'][1]
        self.ROBOT_ORIGIN = tuple(config.get('robot_origin', (0.0, 0.0)))
        
        # Motion constraints
        self.MAX_VELOCITY = config['max_velocity']
        self.MIN_VELOCITY = config['min_velocity']
        self.MAX_ACCELERATION = config['max_acceleration']
        self.MAX_JERK = config['max_jerk']
        
        # Discretization parameters
        self.VELOCITY_RESOLUTION = config['velocity_resolution']
        self.DT = config['dt']
        self.THETA_0_RESOLUTION = config.get('theta_0_resolution', 0.1)
        self.THETA_1_RESOLUTION = config.get('theta_1_resolution', 0.1)
        
        # Environmental settings
        self.CONSIDER_GRAVITY = config.get('consider_gravity', True)
        
    def min_reachable_radius(self) -> float:
        """
        Calculates minimum reachable distance from robot base.
        
        Returns:
            float: Minimum reachable radius in meters
            
        Note:
            This occurs when links are folded back on each other
        """
        return max(self.LINK_1 - self.LINK_2, 0)
        
    def max_reachable_radius(self) -> float:
        """
        Calculates maximum reachable distance from robot base.
        
        Returns:
            float: Maximum reachable radius in meters
            
        Note:
            This occurs when links are fully extended
        """
        return self.LINK_1 + self.LINK_2