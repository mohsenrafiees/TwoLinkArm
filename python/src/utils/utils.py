"""
Utility functions for geometric operations and random geometry generation.

This module provides functions for:
- Generating random points in specified regions
- Creating random obstacles with constraints
- Other geometric utility operations
"""


import numpy as np
from typing import Tuple
from enviroment.obstacles import Obstacle

def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[float, float]:
    """
    Generates a random goal position within an annular region.
    
    Uses uniform distribution for both angle and radius to ensure
    uniform coverage of the annular region.
    
    Args:
        min_radius (float): Minimum distance from origin (meters)
        max_radius (float): Maximum distance from origin (meters)
        
    Returns:
        Tuple[float, float]: Random (x, y) coordinates within specified bounds
        
    Raises:
        ValueError: If min_radius > max_radius or either is negative
        
    Note:
        Coordinates are returned in the robot's reference frame
    """

    if min_radius < 0 or max_radius < 0:
        raise ValueError("Radii must be non-negative")
    if min_radius > max_radius:
        raise ValueError("Minimum radius cannot be greater than maximum radius")
    theta = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(min_radius, max_radius)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

def generate_random_circle_obstacle(workspace_size: float, min_distance: float) -> Obstacle:
    """
    Generates a random rectangular obstacle with workspace constraints.
    
    Creates obstacles that:
    - Maintain minimum clearance from robot base
    - Have size proportional to workspace
    - Stay within workspace bounds
    
    Args:
        workspace_size (float): Size of workspace (meters)
        min_distance (float): Minimum distance from origin (meters)
        
    Returns:
        Obstacle: Randomly generated rectangular obstacle
        
    Raises:
        ValueError: If workspace_size <= 0 or min_distance < 0
    """
    if workspace_size <= 0:
        raise ValueError("Workspace size must be positive")
    if min_distance < 0:
        raise ValueError("Minimum distance must be non-negative")
    if min_distance > workspace_size:
        raise ValueError("Minimum distance cannot exceed workspace size")
    
    max_attempts = 100  # Prevent infinite loops
    for _ in range(max_attempts):
        # Define maximum size for the rectangle
        max_width = workspace_size / 4
        max_height = workspace_size / 4

        # Random width and height
        width = np.random.uniform(workspace_size / 10, max_width)
        height = np.random.uniform(workspace_size / 10, max_height)

        # Random position within workspace
        x_min = -workspace_size / 2 + width
        x_max = workspace_size / 2 - width
        y_min = -workspace_size / 2 + height
        y_max = workspace_size / 2 - height

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # Check distance from robot's base at (0, 0)
        obstacle_center_x = x + width / 2
        obstacle_center_y = y + height / 2
        distance = np.hypot(obstacle_center_x, obstacle_center_y)

        # Consider half of the obstacle's diagonal as its radius
        obstacle_radius = np.hypot(width / 2, height / 2)

        if distance - obstacle_radius >= min_distance:
            return Obstacle(shape='circle', position=(x, y), size=obstacle_radius)

    # If unable to find a suitable position, raise an exception
    raise ValueError("Unable to generate obstacle far enough from the robot's base.")

# def generate_random_circle_obstacle(workspace_size: float, min_distance: float) -> Obstacle:
#     """
#     Generates a random circular obstacle with workspace constraints.
    
#     Creates circular obstacles that:
#     - Maintain minimum clearance from robot base at (150,150)
#     - Have radius proportional to workspace
#     - Stay within workspace bounds
#     - Ensures the entire circle remains in workspace
#     - Considers robot arm reach (125 units from base)
    
#     Args:
#         workspace_size (float): Size of workspace (meters)
#         min_distance (float): Minimum distance from robot's maximum reach (meters)
        
#     Returns:
#         Obstacle: Randomly generated circular obstacle
#     """
#     if workspace_size <= 0:
#         raise ValueError("Workspace size must be positive")
#     if min_distance < 0:
#         raise ValueError("Minimum distance must be non-negative")
#     if min_distance > workspace_size:
#         raise ValueError("Minimum distance cannot exceed workspace size")
        
#     # Robot parameters
#     ROBOT_BASE = np.array([150, 150])
#     ROBOT_MAX_REACH = 125  # Combined length of both arms (75 + 50)
    
#     # Define radius bounds (proportional to workspace)
#     min_radius = workspace_size * 0.10
#     max_radius = workspace_size * 0.2
    
#     # Generate random radius
#     radius = np.random.uniform(min_radius, max_radius)
    
#     # Calculate position bounds to ensure circle stays in workspace
#     position_limit = workspace_size - radius
    
#     max_attempts = 1000
#     attempt = 0
    
#     while attempt < max_attempts:
#         # Generate random position
#         x = np.random.uniform(0, position_limit)
#         y = np.random.uniform(0, position_limit)
        
#         # Calculate distance from robot base
#         distance_from_robot = np.sqrt(
#             (x - ROBOT_BASE[0])**2 + 
#             (y - ROBOT_BASE[1])**2
#         )
        
#         # Check if obstacle is far enough from robot's maximum reach
#         safe_distance = ROBOT_MAX_REACH + radius + min_distance
        
#         if distance_from_robot >= safe_distance:
#             return Obstacle(
#                 shape='circle',
#                 position=(x, y),
#                 size=radius
#             )
            
#         attempt += 1
    
#     raise RuntimeError("Could not find valid obstacle position after maximum attempts")