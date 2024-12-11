from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class State:
    """
    Immutable representation of robot joint positions, velocities, and accelerations.
    
    This class represents the complete state of a 2-DOF robotic arm,
    including joint angles, angular velocities, and angular accelerations.
    Acceleration values are optional and default to 0.0 for backward compatibility.
    
    Attributes:
        theta_0 (float): First joint angle (rad)
        theta_1 (float): Second joint angle (rad)
        omega_0 (float): First joint angular velocity (rad/s)
        omega_1 (float): Second joint angular velocity (rad/s)
        alpha_0 (float, optional): First joint angular acceleration (rad/s^2)
        alpha_1 (float, optional): Second joint angular acceleration (rad/s^2)
        
    Note:
        - States are compared with a tolerance of 1e-3 for floating-point safety
        - Hash values are computed using rounded values for consistency
        - Implements total ordering for use in priority queues
    """
    
    # State variables with physical units
    theta_0: float  # rad
    theta_1: float  # rad
    omega_0: float  # rad/s
    omega_1: float  # rad/s
    alpha_0: float = 0.0  # rad/s^2, optional with default
    alpha_1: float = 0.0  # rad/s^2, optional with default
    
    # Numerical tolerance for floating-point comparisons
    _TOLERANCE: float = 1e-3
    # Precision for hash computation
    _HASH_PRECISION: int = 3
    # Multiplier for hash combination
    _HASH_MULTIPLIER: int = 31

    @classmethod
    def from_partial_state(cls, theta_0: float, theta_1: float, omega_0: float = 0.0, 
                          omega_1: float = 0.0, alpha_0: float = 0.0, alpha_1: float = 0.0) -> 'State':
        """
        Create a State instance with optional parameters defaulting to zero.
        
        Args:
            theta_0: First joint angle (rad)
            theta_1: Second joint angle (rad)
            omega_0: First joint angular velocity (rad/s), defaults to 0
            omega_1: Second joint angular velocity (rad/s), defaults to 0
            alpha_0: First joint angular acceleration (rad/s^2), defaults to 0
            alpha_1: Second joint angular acceleration (rad/s^2), defaults to 0
            
        Returns:
            State: New State instance with specified values
        """
        return cls(theta_0=theta_0, theta_1=theta_1, 
                  omega_0=omega_0, omega_1=omega_1,
                  alpha_0=alpha_0, alpha_1=alpha_1)
    
    def __hash__(self) -> int:
        """
        Computes hash value based on rounded state components.
        """
        h = hash(round(self.theta_0, self._HASH_PRECISION))
        h = self._HASH_MULTIPLIER * h + hash(round(self.theta_1, self._HASH_PRECISION))
        h = self._HASH_MULTIPLIER * h + hash(round(self.omega_0, self._HASH_PRECISION))
        h = self._HASH_MULTIPLIER * h + hash(round(self.omega_1, self._HASH_PRECISION))
        h = self._HASH_MULTIPLIER * h + hash(round(self.alpha_0, self._HASH_PRECISION))
        h = self._HASH_MULTIPLIER * h + hash(round(self.alpha_1, self._HASH_PRECISION))
        return h
    
    def __eq__(self, other) -> bool:
        """
        Checks equality with another state using numerical tolerance.
        """
        if not isinstance(other, State):
            return False
            
        return (abs(self.theta_0 - other.theta_0) < self._TOLERANCE and
                abs(self.theta_1 - other.theta_1) < self._TOLERANCE and
                abs(self.omega_0 - other.omega_0) < self._TOLERANCE and
                abs(self.omega_1 - other.omega_1) < self._TOLERANCE and
                abs(self.alpha_0 - other.alpha_0) < self._TOLERANCE and
                abs(self.alpha_1 - other.alpha_1) < self._TOLERANCE)
    
    def __lt__(self, other: 'State') -> bool:
        """
        Implements less-than comparison for total ordering.
        """
        if not isinstance(other, State):
            raise TypeError("Can only compare with other State instances")
            
        return (self.theta_0, self.theta_1, self.omega_0, self.omega_1, self.alpha_0, self.alpha_1) < (
            other.theta_0, other.theta_1, other.omega_0, other.omega_1, other.alpha_0, other.alpha_1)
    
    def to_dict(self) -> dict:
        """
        Convert state to dictionary representation.
        
        Returns:
            dict: Dictionary containing state variables
        """
        return {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1,
            'omega_0': self.omega_0,
            'omega_1': self.omega_1,
            'alpha_0': self.alpha_0,
            'alpha_1': self.alpha_1
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'State':
        """
        Create State instance from dictionary.
        
        Args:
            data: Dictionary containing state variables
            
        Returns:
            State: New State instance
        """
        return cls(
            theta_0=data['theta_0'],
            theta_1=data['theta_1'],
            omega_0=data.get('omega_0', 0.0),
            omega_1=data.get('omega_1', 0.0),
            alpha_0=data.get('alpha_0', 0.0),
            alpha_1=data.get('alpha_1', 0.0)
        )