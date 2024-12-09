from typing import Tuple, Union

class Obstacle:
    def __init__(self, shape: str, position: Tuple[float, float], size: Union[float, Tuple[float, float]]):
        self.shape = shape  # 'circle' or 'rectangle'
        self.position = position
        self.size = size

    def inflate(self, inflation: float) -> 'Obstacle':
        """
        Creates an inflated version of the obstacle.
        
        Args:
            inflation: Amount to expand the obstacle by
        
        Returns:
            A new Obstacle instance with inflated dimensions
        """
        if self.shape == 'circle':
            # For circles, simply increase the radius
            new_size = self.size + inflation
            
        elif self.shape == 'rectangle':
            # For rectangles, increase both width and height
            width, height = self.size
            # Increase both dimensions
            new_size = (width +  1.4 * inflation, height + 1.4 * inflation)

                
        return Obstacle(self.shape, self.position, new_size)
