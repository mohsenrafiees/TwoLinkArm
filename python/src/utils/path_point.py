from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class PathPoint:
    """
    Represents a point along a parameterized robot path with its derivatives.
    
    This class stores joint positions and their first/second derivatives
    with respect to the path parameter for a 2-DOF robotic arm. It's used
    for trajectory generation and motion planning.
    
    Attributes:
        s (float): Path parameter value in range [0, 1]
        
        theta_0 (float): First joint angle (rad)
        theta_1 (float): Second joint angle (rad)
        
        dtheta_0 (float): First derivative of theta_0 with respect to s 
                         Represents the rate of change of first joint angle
                         along the path
        
        dtheta_1 (float): First derivative of theta_1 with respect to s (rad)
                         Represents the rate of change of second joint angle
                         along the path
        
        ddtheta_0 (float): Second derivative of theta_0 with respect to s 
                          Represents the curvature of first joint angle
                          along the path
        
        ddtheta_1 (float): Second derivative of theta_1 with respect to s
                          Represents the curvature of second joint angle
                          along the path
                          
    Note:
        All derivatives are with respect to the path parameter s, not time.
        To get time derivatives, these values must be multiplied by appropriate
        powers of ds/dt.
    """
    
    s: float
    theta_0: float
    theta_1: float
    dtheta_0: float
    dtheta_1: float
    ddtheta_0: float
    ddtheta_1: float