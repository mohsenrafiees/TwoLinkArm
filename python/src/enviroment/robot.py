from typing import List, Tuple, Union, Dict, Set, Optional
import numpy as np
from utils.constants import RobotConstants
from utils.debug_helper import DebugHelper
from utils.state import State
from enviroment.collision_checker import CollisionChecker

class Robot:
    """
    Implements a two-link robotic arm with full dynamics and kinematics.
    
    This class provides a complete implementation of a 2-DOF robotic arm including:
    - Forward and inverse kinematics
    - Dynamic modeling with optional gravity compensation
    - Joint limit validation
    - Self-collision detection
    - State history tracking
    
    Attributes:
        constants (RobotConstants): Robot physical parameters and limits
        theta_0, theta_1 (float): Joint angles (rad)
        omega_0, omega_1 (float): Joint velocities (rad/s)
        alpha_0, alpha_1 (float): Joint accelerations (rad/s²)
        m1, m2 (float): Link masses (kg)
        I1, I2 (float): Link inertias (kg·m²)
    """

    def __init__(self, constants: RobotConstants) -> None:
        """
        Initializes robot with given physical constants.
        
        Args:
            constants: Physical parameters, limits, and configuration settings
        """
        self.constants = constants
        
        # State histories for analysis and debugging
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.all_omega_0: List[float] = []
        self.all_omega_1: List[float] = []
        self.all_alpha_0: List[float] = []
        self.all_alpha_1: List[float] = []
        
        # Current state variables
        self._theta_0 = 0.0
        self._theta_1 = 0.0
        self.omega_0 = 0.0
        self.omega_1 = 0.0
        self.alpha_0 = 0.0
        self.alpha_1 = 0.0
        
        # Physical parameters
        self.m1 = 0.75  # Mass of link 1 (kg)
        self.m2 = 0.5   # Mass of link 2 (kg)
        # Inertias assuming uniform density links
        self.I1 = self.m1 * constants.LINK_1**2 / 12  
        self.I2 = self.m2 * constants.LINK_2**2 / 12
        
        self.debug_helper = DebugHelper(debug=True)

    @property
    def theta_0(self) -> float:
        """First joint angle in radians."""
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        """
        Sets first joint angle while enforcing limits and recording history.
        
        Args:
            value: Joint angle in radians
            
        Raises:
            ValueError: If joint angle exceeds limits
        """
        self._theta_0 = value
        self.all_theta_0.append(value)
        self._validate_joint_limits(value, 0)

    @property
    def theta_1(self) -> float:
        """Second joint angle in radians."""
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        """
        Sets second joint angle while enforcing limits and recording history.
        
        Args:
            value: Joint angle in radians
            
        Raises:
            ValueError: If joint angle exceeds limits
        """
        self._theta_1 = value
        self.all_theta_1.append(value)
        self._validate_joint_limits(value, 1)

    def _validate_joint_limits(self, theta: float, joint_id: int) -> None:
        """
        Validates that joint angle is within specified limits.
        
        Args:
            theta: Joint angle to validate (rad)
            joint_id: Index of joint (0 or 1)
            
        Raises:
            ValueError: If angle exceeds joint limits
        """
        if not (self.constants.JOINT_LIMITS[0] <= theta <= self.constants.JOINT_LIMITS[1]):
            raise ValueError(f"Joint {joint_id} angle {theta} exceeds joint limits.")

    def joint_1_pos(self, theta_0: Optional[float] = None) -> Tuple[float, float]:
        """
        Computes the (x, y) position of the first joint.
        
        Args:
            theta_0: Optional joint angle (rad). Uses current angle if None
            
        Returns:
            Tuple[float, float]: (x, y) coordinates of joint 1
        """
        if theta_0 is None:
            theta_0 = self.theta_0
        return (
            self.constants.LINK_1 * np.cos(theta_0),
            self.constants.LINK_1 * np.sin(theta_0)
        )

    def joint_2_pos(self, theta_0: Optional[float] = None, theta_1: Optional[float] = None) -> Tuple[float, float]:
        """
        Computes the (x, y) position of the end-effector.
        
        Args:
            theta_0: Optional first joint angle (rad). Uses current if None
            theta_1: Optional second joint angle (rad). Uses current if None
            
        Returns:
            Tuple[float, float]: (x, y) coordinates of end-effector
        """
        if theta_0 is None:
            theta_0 = self.theta_0
        if theta_1 is None:
            theta_1 = self.theta_1
        return self.forward_kinematics(theta_0, theta_1)

    def compute_link_segments(self, theta_0: float, theta_1: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Computes line segments representing the robot's links.
        
        Args:
            theta_0: First joint angle (rad)
            theta_1: Second joint angle (rad)
            
        Returns:
            List of line segments as ((x1,y1), (x2,y2)) tuples
        """
        j0 = self.constants.ROBOT_ORIGIN
        j1 = self.joint_1_pos(theta_0)
        j2 = self.forward_kinematics(theta_0, theta_1)
        return [(j0, j1), (j1, j2)]

    def forward_kinematics(self, theta_0: float, theta_1: float) -> Tuple[float, float]:
        """
        Computes end-effector position given joint angles.
        
        Args:
            theta_0: First joint angle (rad)
            theta_1: Second joint angle (rad)
            
        Returns:
            Tuple[float, float]: (x, y) coordinates of end-effector
        """
        x = (self.constants.LINK_1 * np.cos(theta_0) + 
             self.constants.LINK_2 * np.cos(theta_0 + theta_1))
        y = (self.constants.LINK_1 * np.sin(theta_0) + 
             self.constants.LINK_2 * np.sin(theta_0 + theta_1))
        return x, y

    def inverse_kinematics(self, x: float, y: float) -> Tuple[float, float]:
        """
        Computes joint angles given desired end-effector position.
        
        Uses geometric approach and selects solution closest to current configuration.
        
        Args:
            x: Desired x-coordinate
            y: Desired y-coordinate
            
        Returns:
            Tuple[float, float]: Joint angles (theta_0, theta_1) in radians
            
        Note:
            Returns solution closest to current joint angles when multiple exist
        """
        # Compute theta_1 using cosine law
        cos_theta_1 = ((x**2 + y**2 - self.constants.LINK_1**2 - self.constants.LINK_2**2)
                      / (2 * self.constants.LINK_1 * self.constants.LINK_2))
        cos_theta_1 = np.clip(cos_theta_1, -1.0, 1.0)
        
        # Get both possible solutions
        theta_1_options = [np.arccos(cos_theta_1), -np.arccos(cos_theta_1)]
        solutions = []
        
        for theta_1 in theta_1_options:
            k1 = self.constants.LINK_1 + self.constants.LINK_2 * np.cos(theta_1)
            k2 = self.constants.LINK_2 * np.sin(theta_1)
            theta_0 = np.arctan2(y, x) - np.arctan2(k2, k1)
            solutions.append((theta_0, theta_1))

        # Select solution closest to current configuration
        current_theta_0, current_theta_1 = self.theta_0, self.theta_1
        min_distance = float('inf')
        best_solution = solutions[0]
        
        for solution in solutions:
            distance = abs(solution[0] - current_theta_0) + abs(solution[1] - current_theta_1)
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
                
        return best_solution

    def compute_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes dynamic matrices for robot motion equations.
        
        Calculates mass matrix (M), Coriolis matrix (C), and gravity vector (G)
        for equations of motion: M * ddq + C * dq + G = tau
        
        Args:
            theta_0, theta_1: Joint angles (rad)
            omega_0, omega_1: Joint velocities (rad/s)
            
        Returns:
            Tuple of arrays (M, C, G):
            - M: 2x2 mass matrix
            - C: 2x2 Coriolis matrix
            - G: 2x1 gravity vector (zeros if gravity disabled)
        """
        l1, l2 = self.constants.LINK_1, self.constants.LINK_2
        g = 9.81 if self.constants.CONSIDER_GRAVITY else 0.0
        
        # Mass matrix
        M11 = (self.m1*l1**2/4 + self.I1 + 
               self.m2*(l1**2 + l2**2/4 + l1*l2*np.cos(theta_1)))
        M12 = self.m2 * (l2**2/4 + l1*l2*np.cos(theta_1)/2)
        M21 = M12
        M22 = self.m2 * l2**2/4 + self.I2
        M = np.array([[M11, M12], [M21, M22]])
        
        # Coriolis matrix
        h = self.m2 * l1 * l2 * np.sin(theta_1)
        C = np.array([
            [-h * omega_1, -h * (omega_0 + omega_1)],
            [h * omega_0, 0]
        ])
        
        # Gravity vector
        if self.constants.CONSIDER_GRAVITY:
            G1 = ((self.m1*l1/2 + self.m2*l1)*g*np.cos(theta_0) + 
                  self.m2*l2*g*np.cos(theta_0 + theta_1)/2)
            G2 = self.m2*l2*g*np.cos(theta_0 + theta_1)/2
        else:
            G1, G2 = 0.0, 0.0
        G = np.array([G1, G2])
        
        return M, C, G
        
    def forward_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float,
                        tau_0: float, tau_1: float) -> Tuple[float, float]:
        """
        Computes joint accelerations given applied torques.
        
        Args:
            theta_0, theta_1: Joint angles (rad)
            omega_0, omega_1: Joint velocities (rad/s)
            tau_0, tau_1: Applied joint torques (N⋅m)
            
        Returns:
            Tuple[float, float]: Joint accelerations (rad/s²)
        """
        M, C, G = self.compute_dynamics(theta_0, theta_1, omega_0, omega_1)
        omega = np.array([omega_0, omega_1])
        tau = np.array([tau_0, tau_1])
        
        # Solve M * alpha = tau - C * omega - G
        alpha = np.linalg.solve(M, tau - C @ omega - G)
        return alpha[0], alpha[1]
        
    def inverse_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float,
                        alpha_0: float, alpha_1: float) -> Tuple[float, float]:
        """
        Computes required torques for desired accelerations.
        
        Args:
            theta_0, theta_1: Joint angles (rad)
            omega_0, omega_1: Joint velocities (rad/s)
            alpha_0, alpha_1: Desired joint accelerations (rad/s²)
            
        Returns:
            Tuple[float, float]: Required joint torques (N⋅m)
        """
        M, C, G = self.compute_dynamics(theta_0, theta_1, omega_0, omega_1)
        omega = np.array([omega_0, omega_1])
        alpha = np.array([alpha_0, alpha_1])
        
        # tau = M * alpha + C * omega + G
        tau = M @ alpha + C @ omega + G
        return tau[0], tau[1]

    def self_collision(self, theta_0: float, theta_1: float) -> bool:
        """
        Checks for self-collision between robot links and base.
        
        Args:
            theta_0: First joint angle (rad)
            theta_1: Second joint angle (rad)
            
        Returns:
            bool: True if self-collision detected, False otherwise
        """
        # Get joint positions
        base = self.constants.ROBOT_ORIGIN
        joint1 = self.joint_1_pos(theta_0)
        joint2 = self.joint_2_pos(theta_0, theta_1)

        # Check second link collision with base
        return CollisionChecker.line_circle_collision(joint1, joint2, base, 
                                                    self.constants.BASE_RADIUS)