from typing import List, Tuple, Optional
import numpy as np
from enviroment.robot import Robot
from utils.constants import RobotConstants
from utils.debug_helper import DebugHelper
from utils.state import State

class MPCController:
    """
    Model Predictive Controller for 2-link robotic arm. Based on following paper
    https://www.mdpi.com/2075-1702/6/3/37
    
    This controller implements a predictive control scheme that optimizes robot motion
    over a time horizon while respecting dynamic constraints. It uses feedback
    linearization for computing control inputs and includes trajectory tracking.
    
    Attributes:
        h (float): Prediction horizon in seconds
        dt (float): Control timestep in seconds
        zeta (float): Damping ratio for control response
        w0 (float): Natural frequency for control response (rad/s)
        Note: Selection of h and wo depends on actuator torque
    """
    
    def __init__(self, robot: Optional[Robot], constants: RobotConstants) -> None:
        """
        Initialize the MPC controller.

        Args:
            robot: Robot instance containing dynamic model (can be None initially)
            constants: Robot constants containing physical parameters and limits
        """
        self.robot = robot
        self.constants = constants
        self.h = 0.1716        # Prediction horizon (s)(0.1716 is more aggressive)
        self.dt = constants.DT   # Control timestep (s)
        self.zeta = 0.9          # Damping ratio (dimensionless)
        self.w0 = 4           # Natural frequency (rad/s) (4.0 is more aggressive)
        
        # Reference trajectory storage
        self.path: List[State] = []
        self.path_index = 0
        
        # Initialize debug helper
        self.debug_helper = DebugHelper(debug=True)
        
        # Compute controller gains using optimal control formulation
        self.k1 = 2 / (self.h**2 + 4*self.get_rho())           # Position gain
        self.k2 = (2*self.h**2 + 4*self.get_rho()) / \
                 (self.h**3 + 4*self.get_rho()*self.h)        # Velocity gain
        
        # State history for analysis and debugging
        self.theta_ref_history: List[Tuple[float, float]] = []
        self.theta_actual_history: List[Tuple[float, float]] = []
        self.v_history: List[Tuple[float, float]] = []
        self.tau_history: List[Tuple[float, float]] = []

    def get_rho(self) -> float:
        """
        Compute weight factor ρ for optimal control formulation.

        The weight factor balances tracking performance and control effort based on
        natural frequency and prediction horizon.

        Returns:
            float: Computed weight factor ρ
        """
        w0h = self.w0 * self.h
        rho = (2 - w0h**2) / (4 * self.w0**2)
        return rho

    def compute_feedback_linearization(self, theta: Tuple[float, float], 
                                     theta_dot: Tuple[float, float],
                                     v: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute control torques using feedback linearization.

        Args:
            theta: Current joint angles (θ₀, θ₁) in radians
            theta_dot: Current joint velocities (ω₀, ω₁) in rad/s
            v: Synthetic control inputs (v₀, v₁)

        Returns:
            Tuple[float, float]: Computed joint torques (τ₀, τ₁) in N⋅m
        """
        # Get dynamic matrices
        M, C, G = self.robot.compute_dynamics(theta[0], theta[1], 
                                            theta_dot[0], theta_dot[1])
        
        # Compute torques: τ = M(θ)v + C(θ,θ̇)θ̇ + G(θ)
        tau = M @ np.array(v) + C @ np.array(theta_dot) + G
        
        return tau[0], tau[1]

    def compute_synthetic_control(self, theta_ref: Tuple[float, float],
                                theta: Tuple[float, float],
                                theta_dot: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute synthetic control inputs using optimal gains.

        Args:
            theta_ref: Reference joint angles (θᵣ₀, θᵣ₁) in radians
            theta: Current joint angles (θ₀, θ₁) in radians
            theta_dot: Current joint velocities (ω₀, ω₁) in rad/s

        Returns:
            Tuple[float, float]: Synthetic control inputs (v₀, v₁)
        """
        # Compute control law: v = k₁(θᵣ - θ) - k₂θ̇
        # Compute errors
        v1 = self.k1 * (theta_ref[0] - theta[0]) - self.k2 * theta_dot[0]
        v2 = self.k1 * (theta_ref[1] - theta[1]) - self.k2 * theta_dot[1]
        
        return v1, v2

    def optimize_control_sequence(self, current_state: Tuple[float, float, float, float],
                                reference_trajectory: List[Tuple[float, float]], 
                                steps_in_horizon: int) -> List[Tuple[float, float]]:
        """
        Optimize control sequence over prediction horizon.

        Args:
            current_state: Current state (θ₀, θ₁, ω₀, ω₁)
            reference_trajectory: List of reference states over horizon
            steps_in_horizon: Number of timesteps in prediction horizon

        Returns:
            List[Tuple[float, float]]: Optimized control sequence
        """
        control_sequence = []
        
        for i in range(steps_in_horizon):
            # Get reference state for this step
            ref_i = reference_trajectory[min(i, len(reference_trajectory) - 1)]
            
            # Compute optimal control
            v = self.compute_synthetic_control(
                ref_i,
                (current_state[0], current_state[1]),
                (current_state[2], current_state[3])
            )
            
            control_sequence.append(v)
            
        return control_sequence

    def step(self, theta_ref: Tuple[float, float], robot: Robot, 
             is_moving_goal: bool) -> Tuple[float, float]:
        """
        Execute one control step.

        Args:
            theta_ref: Reference joint angles (θᵣ₀, θᵣ₁) in radians
            robot: Current robot state
            is_moving_goal: Flag indicating if target is moving

        Returns:
            Tuple[float, float]: Control torques (τ₀, τ₁) in N⋅m
        """
        # Get current state
        current_state = (robot.theta_0, robot.theta_1, robot.omega_0, robot.omega_1)
        
        # Generate reference trajectory
        steps_in_horizon = int(self.h / self.dt)
        reference_trajectory = self._generate_reference_trajectory(steps_in_horizon)
        
        # Optimize control sequence
        control_sequence = self.optimize_control_sequence(
            current_state, reference_trajectory, steps_in_horizon)
        
        # Get control action
        ref_state = self.path[self.path_index]
        v =  control_sequence[0]
        
        # Compute and apply torques
        tau = self.compute_feedback_linearization(
            (robot.theta_0, robot.theta_1),
            (robot.omega_0, robot.omega_1),
            v
        )
        
        # Update history
        self._update_history(theta_ref, robot, v, tau)
        
        return tau

    def reset(self) -> None:
        """Reset controller state and clear all history."""
        self.theta_ref_history.clear()
        self.theta_actual_history.clear()
        self.v_history.clear()
        self.tau_history.clear()

    def set_path(self, path: List[State]) -> None:
        """
        Set new reference path for controller.

        Args:
            path: List of states defining reference trajectory
        """
        self.path = path
        self.path_index = 0

    def set_path_index(self, path_index: int) -> None:
        """
        Set current position in reference path.

        Args:
            path_index: New path index
        """
        self.path_index = path_index

    # def get_current_path_index(self) -> int:
    #     """
    #     Get current path index with bounds checking.

    #     Returns:
    #         int: Current valid path index
    #     """
    #     return min(self.path_index, len(self.path) - 1) if self.path else 0

    #     current_pos = robot.joint_2_pos()
    #     look_ahead = min(1, len(self.path) - self.path_index)
    #     min_dist = float('inf')
    #     best_idx = self.path_index
        
    #     # Find closest point within look-ahead window
    #     for i in range(self.path_index, self.path_index + look_ahead):
    #         if i >= len(self.path):
    #             break
                
    #         path_pos = robot.forward_kinematics(
    #             self.path[i].theta_0,
    #             self.path[i].theta_1
    #         )
            
    #         dist = np.hypot(current_pos[0] - path_pos[0],
    #                       current_pos[1] - path_pos[1])
            
    #         if dist < min_dist:
    #             min_dist = dist
    #             best_idx = i
        
    #     # Update index if making forward progress
    #     if best_idx > self.path_index:
    #         self.path_index = best_idx

    def _log_initialization(self) -> None:
        """Log controller initialization parameters."""
        self.debug_helper.log_state("\nMPC Controller Initialized:")
        self.debug_helper.log_state(f"Horizon: {self.h}, DT: {self.dt}")
        self.debug_helper.log_state(f"Controller gains - k1: {self.k1:.3f}, k2: {self.k2:.3f}")

    def _generate_reference_trajectory(self, steps_in_horizon: int) -> List[Tuple[float, float]]:
        """
        Generate reference trajectory over prediction horizon.

        Args:
            steps_in_horizon: Number of timesteps in horizon

        Returns:
            List[Tuple[float, float]]: Reference trajectory points
        """
        reference_trajectory = []
        current_path_index = self.path_index #self.get_current_path_index()
        
        for i in range(steps_in_horizon):
            path_idx = min(current_path_index + i, len(self.path) - 1)
            
            if path_idx < len(self.path):
                ref_state = self.path[path_idx]
                reference_trajectory.append((ref_state.theta_0, ref_state.theta_1))
                
        return reference_trajectory

    def _should_stop(self, is_moving_goal: bool, ref_state: State) -> bool:
        """
        Determine if robot should stop based on current state.

        Args:
            is_moving_goal: Flag indicating if target is moving
            ref_state: Current reference state

        Returns:
            bool: True if robot should stop, False otherwise
        """
        return (is_moving_goal and self.path_index > 3 and
                ref_state.omega_0 == 0.0 and ref_state.omega_1 == 0.0)

    def _update_history(self, theta_ref: Tuple[float, float], robot: Robot,
                       v: Tuple[float, float], tau: Tuple[float, float]) -> None:
        """
        Update controller state history.

        Args:
            theta_ref: Reference joint angles
            robot: Current robot state
            v: Synthetic control inputs
            tau: Applied torques
        """
        self.theta_ref_history.append(theta_ref)
        self.theta_actual_history.append((robot.theta_0, robot.theta_1))
        self.v_history.append(v)
        self.tau_history.append(tau)