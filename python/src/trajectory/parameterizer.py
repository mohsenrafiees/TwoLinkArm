from typing import List
import numpy as np
import scipy.interpolate
from utils.state import State
from utils.path_point import PathPoint
from utils.debug_helper import DebugHelper

class PathParameterizer:
    """
    Converts discrete path points to a continuous path with derivatives.
    
    This class takes discrete robot joint states and creates continuous spline 
    interpolations for positions and velocities. It enables querying the path
    at arbitrary parameter values while maintaining smoothness.
    
    Attributes:
        original_points (List[State]): Original input path points
        s_points (np.ndarray): Uniformly spaced path parameter values
        theta_0_spline (scipy.interpolate.CubicSpline): Spline for joint 0 position
        theta_1_spline (scipy.interpolate.CubicSpline): Spline for joint 1 position
        omega_0_spline (scipy.interpolate.CubicSpline): Spline for joint 0 velocity
        omega_1_spline (scipy.interpolate.CubicSpline): Spline for joint 1 velocity
    """
    
    def __init__(self, path_points: List[State]):
        """
        Initializes path parameterizer with discrete path points.
        
        Args:
            path_points (List[State]): List of robot states containing joint positions
                                     and velocities
                                     
        Raises:
            ValueError: If path_points is empty or contains fewer than 2 points
        """
        if not path_points:
            raise ValueError("Path points list cannot be empty")
        if len(path_points) < 2:
            raise ValueError("Need at least 2 path points for interpolation")
            
        self.original_points = path_points
        
        # Create dense sampling for smooth interpolation
        num_points = max(len(path_points) * 2, 200)
        self.s_points = np.linspace(0, 1, num_points)
        
        # Extract joint positions and velocities
        self.theta_0_points = [p.theta_0 for p in path_points]
        self.theta_1_points = [p.theta_1 for p in path_points]
        self.omega_0_points = [p.omega_0 for p in path_points]
        self.omega_1_points = [p.omega_1 for p in path_points]
        
        # Create normalized path parameter values
        original_s = np.linspace(0, 1, len(path_points))
        
        # Initialize splines with clamped boundary conditions for smooth derivatives
        self.theta_0_spline = scipy.interpolate.CubicSpline(
            original_s, self.theta_0_points, 
            bc_type='clamped'  # Ensures continuous derivatives at boundaries
        )
        self.theta_1_spline = scipy.interpolate.CubicSpline(
            original_s, self.theta_1_points,
            bc_type='clamped'
        )
        
        self.omega_0_spline = scipy.interpolate.CubicSpline(
            original_s, self.omega_0_points,
            bc_type='clamped'
        )
        self.omega_1_spline = scipy.interpolate.CubicSpline(
            original_s, self.omega_1_points,
            bc_type='clamped'
        )

    def get_path_point(self, s: float) -> PathPoint:
        """
        Computes path point and its derivatives at given path parameter value.
        
        Args:
            s (float): Path parameter value in range [0, 1]
            
        Returns:
            PathPoint: Path point containing positions and first/second derivatives
                      for both joints
                      
        Note:
            Input s is automatically clipped to valid range [0, 1]
        """
        s = np.clip(s, 0, 1)
            
        # Compute joint positions
        theta_0 = float(self.theta_0_spline(s))
        theta_1 = float(self.theta_1_spline(s))
        
        # Compute first derivatives (dtheta/ds)
        dtheta_0 = float(self.theta_0_spline.derivative(1)(s))
        dtheta_1 = float(self.theta_1_spline.derivative(1)(s))
        
        # Compute second derivatives (d²theta/ds²)
        ddtheta_0 = float(self.theta_0_spline.derivative(2)(s))
        ddtheta_1 = float(self.theta_1_spline.derivative(2)(s))
        
        return PathPoint(s, theta_0, theta_1, dtheta_0, dtheta_1, ddtheta_0, ddtheta_1)