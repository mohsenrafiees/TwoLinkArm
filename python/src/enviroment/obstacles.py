from dataclasses import dataclass
from typing import Tuple, Union, List, Literal
import numpy as np

class Obstacle:
    """
    Represents a 2D obstacle in the robot's workspace.
    
    This class handles both circular and rectangular obstacles, providing
    methods for obstacle inflation used in collision checking and path planning.
    
    Attributes:
        shape (Literal['circle', 'rectangle']): Geometric shape of the obstacle
        position (Tuple[float, float]): (x, y) coordinates of obstacle center
        size (Union[float, Tuple[float, float]]): Dimensions of the obstacle:
            - For circles: radius as float
            - For rectangles: (width, height) as tuple
    """
    
    def __init__(self, 
                 shape: Literal['circle', 'rectangle'], 
                 position: Tuple[float, float], 
                 size: Union[float, Tuple[float, float]]):
        """
        Initializes an obstacle with specified geometry and position.
        
        Args:
            shape: Type of obstacle ('circle' or 'rectangle')
            position: (x, y) coordinates of obstacle center
            size: Dimensions of obstacle (radius for circle, (width, height) for rectangle)
            
        Raises:
            ValueError: If shape is not 'circle' or 'rectangle'
        """
        if shape not in ['circle', 'rectangle']:
            raise ValueError("Shape must be either 'circle' or 'rectangle'")
            
        self.shape = shape
        self.position = position
        self.size = size
        
    def inflate(self, inflation: float) -> 'Obstacle':
        """
        Creates an inflated version of the obstacle for safety margins.
        
        Inflation is applied differently based on obstacle shape:
        - Circles: Radius is increased by inflation amount
        - Rectangles: Both dimensions are increased by √2 * inflation to account
          for corner distances
        
        Args:
            inflation (float): Positive value to expand obstacle dimensions
            
        Returns:
            Obstacle: New instance with inflated dimensions
            
        Note:
            Rectangle inflation uses 1.4 (≈√2) factor to maintain consistent
            safety margins at corners
        """
        if inflation < 0:
            raise ValueError("Inflation amount must be non-negative")
            
        if self.shape == 'circle':
            new_size = self.size + inflation
        elif self.shape == 'rectangle':
            width, height = self.size
            # √2 factor accounts for corner distances
            new_size = (width + 1.4 * inflation, height + 1.4 * inflation)
                
        return Obstacle(self.shape, self.position, new_size)