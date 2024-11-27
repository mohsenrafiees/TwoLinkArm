import numpy as np
from typing import Tuple
from obstacle import Obstacle 

def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[float, float]:
    """
    Generates a random goal within the robot's reachable radius.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(min_radius, max_radius)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

def generate_random_rectangle_obstacle(workspace_size: float, min_distance: float) -> Obstacle:
    """
    Generates a random rectangle obstacle within the workspace, ensuring it's far enough from the robot's base.
    """
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
            return Obstacle('rectangle', (x, y), (width, height))

    # If unable to find a suitable position, raise an exception
    raise ValueError("Unable to generate obstacle far enough from the robot's base.")
