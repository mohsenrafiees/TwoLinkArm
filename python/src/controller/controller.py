from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np
from utils.state import State
from utils.debug_helper import DebugHelper
from controller.mpc_controller import MPCController
from trajectory.generator import TimeOptimalTrajectoryGenerator
import time
import math

if TYPE_CHECKING:
    from enviroment.robot import Robot
    from enviroment.world import World
    from planner.base_planner import Planner

class Controller:
    """
    Robot motion controller implementing Model Predictive Control (MPC) with trajectory tracking.
    
    This controller manages path planning, trajectory generation, and real-time control
    of a robot system. It handles graceful stopping, dynamic replanning, and maintains
    tracking history for debugging and analysis.
    """
    
    def __init__(self, robot: 'Robot', world: 'World', planner: 'Planner') -> None:
        """
        Initialize the controller with robot, world, and planner instances.

        Args:
            robot: Robot instance containing kinematics and dynamics models
            world: World instance containing environment information
            planner: Path planner instance for generating collision-free paths
        """
        # Core components
        self.constants = robot.constants
        self.world = world
        self.planner = planner
        
        # Path and planning variables
        self.path: List[State] = []         # Current trajectory
        self.coarse_path: List[State] = []  # Initial path before optimization
        self.path_index = 0                 # Current position in trajectory
        self.time_index = 0
        self.planning_timeout = 500.0       # Maximum planning time (ms)
        
        # Goal handling
        self.alternative_goal = None        # Backup goal if primary is unreachable
        self.is_alternative = False         # Flag for using alternative goal
        self.is_moving_goal = False         # Flag for tracking moving targets
        self.goal_distance = float('inf')	# Min distance of trajectory to goal
        
        # Stopping behavior
        self.stopping_mode = False          # Flag for graceful stop
        self.stop_start_index = None        # Index where stop was initiated
        self.stop_distance = None           # Calculated stopping distance
        self.current_velocity = 0.0         # Current end-effector velocity
        
        # Initialize MPC controller
        self.mpc = MPCController(None, robot.constants)
        
        # Initialize trajectory tracking arrays
        self._init_tracking_arrays()
        
        # Debug helper
        self.debug_helper = DebugHelper(debug=True)

    def step(self, robot: 'Robot') -> 'Robot':
        """
	       	Execute one control step updating robot state.

        Args:
            robot: Current robot state

        Returns:
            Robot: Updated robot state after control action
        """
        if not self.path or self.path_index >= len(self.path):
            return self._handle_empty_path(robot)
            
        try:
            # Get reference state and compute control
            self.time_index += 1
            ref_state = self.path[self.path_index]
            theta_ref = (ref_state.theta_0, ref_state.theta_1)
            tau_0, tau_1 = self.mpc.step(theta_ref, robot, self.is_moving_goal)
            
            # Update robot dynamics
            alpha_0, alpha_1 = robot.forward_dynamics(
                robot.theta_0, robot.theta_1,
                robot.omega_0, robot.omega_1,
                tau_0, tau_1
            )
            
            # Integrate velocities and positions
            robot.omega_0 += alpha_0 * self.constants.DT
            robot.omega_1 += alpha_1 * self.constants.DT
            robot.theta_0 += robot.omega_0 * self.constants.DT
            robot.theta_1 += robot.omega_1 * self.constants.DT

            # Update tracking history and path progress
            self._update_tracking_history(robot)
            self._update_path_index(robot)
            
            # Periodic debugging
            if len(self.actual_theta_0) % 1000 == 0:
                self._perform_debug_analysis(robot)
            
        except Exception as e:
            self.debug_helper.log_state(f"Controller error: {str(e)}")
            return self._handle_error(robot)
            
        return robot

    def initiate_graceful_stop(self, robot: 'Robot') -> None:
        """
        Initiate a controlled deceleration to stop the robot smoothly.

        Args:
            robot: Current robot state
        """
        # self.debug_helper.log_state("Starting graceful stop")
        
        # # Calculate current velocity and stopping distance
        # self.current_velocity = np.hypot(robot.omega_0, robot.omega_1)
        # self.stop_distance = (self.current_velocity ** 2) / (
        #     2 * 0.85 * self.constants.MAX_ACCELERATION)
        
        # # Mark stop position and calculate points needed
        # self.stop_start_index = self.path_index
        # points_covered = self._calculate_stop_points(robot)
        
        # # Override trajectory velocities for stopping
        # if points_covered > 0:
        #     self._override_trajectory_velocities(points_covered)
        self.stopping_mode = True

    def _init_tracking_arrays(self) -> None:
        """Initialize arrays for tracking reference and actual trajectories."""
        # Reference trajectories
        self.reference_theta_0: List[float] = []
        self.reference_theta_1: List[float] = []
        self.reference_omega_0: List[float] = []
        self.reference_omega_1: List[float] = []
        self.reference_alpha_0: List[float] = []
        self.reference_alpha_1: List[float] = []
        
        # Actual trajectories
        self.actual_theta_0: List[float] = []
        self.actual_theta_1: List[float] = []
        self.actual_omega_0: List[float] = []
        self.actual_omega_1: List[float] = []
        self.actual_alpha_0: List[float] = []
        self.actual_alpha_1: List[float] = []

    def set_goal(self, robot: 'Robot', goal: Tuple[float, float], final_velocities: Tuple[float, float] = (0.0, 0.0)) -> bool:
        """
        Set a new goal position and plan path to reach it.
        Args:
            robot: Robot instance for current state
            goal: Target position in workspace coordinates (x, y)
            final_velocities: Desired final joint velocities (omega0, omega1)
        Returns:
            bool: True if path planning succeeded, False otherwise
        """
        self.planning_mode = True
        self.plan_start_time = time.time()
        
        try:  
            # Check if goal is obstructed and find alternative if needed
            actual_goal = self.find_closest_unobstructed_goal(robot, goal)
            if actual_goal != goal:
                self.debug_helper.log_state(
                    f"Original goal obstructed. Using alternative position: {actual_goal}")
            
            # Get start position and plan path
            start_pos = robot.joint_2_pos()
            self.debug_helper.log_state(f"Planning from {start_pos} to {goal}")
            self.debug_helper.log_state("<======== Path Planning Started ========>")
            coarse_path = self.planner.Plan(start_pos, actual_goal, final_velocities)

            if len(coarse_path) < 2:
                self.debug_helper.log_state("Failed to generate Path")
                self.planning_mode = False
                return False

            # Store coarse path and debug info
            self.coarse_path = coarse_path
            self.debug_helper.print_path_points(coarse_path)
            self.debug_helper.print_path_stats(coarse_path, robot)
            self.debug_helper.validate_path_limits(coarse_path, robot)
            
            # Generate time-optimal trajectory
            self.mpc.robot = robot
            trajectory_planner = TimeOptimalTrajectoryGenerator(robot, coarse_path)
            self.debug_helper.log_state("<======== Trajectory Generation Started ========>")
            path = trajectory_planner.generate_trajectory(coarse_path)
            self.debug_helper.print_trajectory_points(path)
            self.debug_helper.print_trajectory_stats(path, robot)
            
            if not path:
                # Calculate distance between goal and last path point
                last_pose = start_pos
                self.goal_distance = ((actual_goal[0] - last_pose[0])**2 + 
                                   (actual_goal[1] - last_pose[1])**2)**0.5
                self.debug_helper.log_state("Failed to generate trajectory")
                self.planning_mode = False
                return False
                
            # Calculate distance between goal and last path point
            last_point = path[-1]
            last_pose = robot.forward_kinematics(
                last_point.theta_0,
                last_point.theta_1)
            
            self.goal_distance = ((actual_goal[0] - last_pose[0])**2 + 
                               (actual_goal[1] - last_pose[1])**2)**0.5
            
            # Update controller state
            self.path = path
            self.mpc.set_path(path)
            self.path_index = 0
            self._reset_controller_states()
            
            self.debug_helper.analyze_path_tracking(self, robot)
            self.planning_mode = False
            return True
            
        except Exception as e:
            self.debug_helper.log_state(f"Error in path planning: {str(e)}")
            self.planning_mode = False
            return False

    def find_closest_unobstructed_goal(self, robot: 'Robot', goal: Tuple[float, float], 
                                search_radius: float = 1.0, num_candidates: int = 16) -> Tuple[float, float]:
        """
        Find the closest unobstructed goal position by sampling points in a circle around the original goal.
        """
        # First check if original goal is unobstructed
        goal_theta = robot.inverse_kinematics(goal[0], goal[1])
        goal_state = State(goal_theta[0], goal_theta[1], 0.0, 0.0)

        if goal_theta and not self.planner.is_collision(goal_state):
            return goal
        
        # Generate candidate positions in a circle around the goal
        candidates = []
        for i in range(num_candidates):
            angle = 2 * math.pi * i / num_candidates
            x = goal[0] + search_radius * math.cos(angle)
            y = goal[1] + search_radius * math.sin(angle)
        
            # Check if position is reachable and unobstructed
            state_theta = robot.inverse_kinematics(x, y)
            if state_theta:  # Check if inverse kinematics returned a valid solution
                state = State(state_theta[0], state_theta[1], 0.0, 0.0)
                if not self.planner.is_collision(state):
                    # Calculate distance to original goal
                    dist = ((x - goal[0])**2 + (y - goal[1])**2)**0.5
                    candidates.append((dist, (x, y)))
    
        # Sort by distance and return closest valid position
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        elif search_radius < 50.0:  # Try with larger search radius
            return self.find_closest_unobstructed_goal(
                robot, goal, search_radius * 1.5, num_candidates)
        else:
            return goal  # Return original goal if no valid alternative found

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-π, π] range.

        Args:
            angle: Input angle in radians

        Returns:
            float: Normalized angle in radians
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    
    # TODO: Implement a robust algorithm
    def _update_path_index(self, robot: 'Robot') -> None:
        """
        Update path index using a simplified approach that ensures quick updates
        and allows robot to reach target velocities.
        
        Args:
            robot: Current robot state containing position and joint velocities
        """
        try:
            if not self.path or self.path_index >= len(self.path):
                return
                
            current_pos = robot.joint_2_pos()
            
            # Get current reference point position
            current_ref_pos = robot.forward_kinematics(
                self.path[self.path_index].theta_0,
                self.path[self.path_index].theta_1
            )
            
            # Distance to current reference point
            dist_to_current = np.hypot(
                current_pos[0] - current_ref_pos[0],
                current_pos[1] - current_ref_pos[1]
            )
            
            # Look ahead a few points to find better reference
            look_ahead = min(3, len(self.path) - self.path_index)
            best_idx = self.path_index
            min_dist = dist_to_current
            
            for i in range(self.path_index + 1, self.path_index + look_ahead):
                if i >= len(self.path):
                    break
                    
                path_pos = robot.forward_kinematics(
                    self.path[i].theta_0,
                    self.path[i].theta_1
                )
                
                dist = np.hypot(
                    current_pos[0] - path_pos[0],
                    current_pos[1] - path_pos[1]
                )
                
                # Update if we find a closer point
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            # Update index if we found a better point or if we're close enough to current
            if best_idx > self.path_index or self.time_index > self.path_index:
                self.path_index = min(best_idx + 1, len(self.path) - 1)
                self.mpc.set_path_index(self.path_index)                
        except Exception as e:
            self.debug_helper.log_state(f"Error in path index update: {str(e)}")
            self.path_index = min(self.path_index + 1, len(self.path) - 1)
            self.mpc.set_path_index(self.path_index)
    # def _update_path_index(self, robot: 'Robot') -> None:
        """
        Update path index to next point when robot is close enough to current point.
        Maintains current index if robot is too far from current point.
        
        Args:
            robot: Current robot state containing position
        """
        # try:
        #     if not self.path or self.path_index >= len(self.path) - 1:  # Check if we're at end
        #         return
                
        #     current_pos = robot.joint_2_pos()
        #     current_path_pos = robot.forward_kinematics(
        #         self.path[self.path_index].theta_0,
        #         self.path[self.path_index].theta_1
        #     )



        #     current_pos = robot.joint_2_pos()
        #     current_path_pos = robot.forward_kinematics(
        #         self.path[self.path_index].theta_0,
        #         self.path[self.path_index].theta_1
        #     )
            
        #     # Get vector from current position to path point
        #     path_vector = np.array([
        #         current_path_pos[0] - current_pos[0],
        #         current_path_pos[1] - current_pos[1]
        #     ])
            
           
                
        #     # Compute end-effector velocity using Jacobian
        #     th0, th1 = robot.theta_0, robot.theta_1
        #     l1, l2 = robot.constants.LINK_1, robot.constants.LINK_2
            
        #     J = np.array([
        #         [-l1*np.sin(th0) - l2*np.sin(th0 + th1), -l2*np.sin(th0 + th1)],
        #         [l1*np.cos(th0) + l2*np.cos(th0 + th1), l2*np.cos(th0 + th1)]
        #     ])
            
        #     joint_velocities = np.array([robot.omega_0, robot.omega_1])
        #     ee_velocity = J @ joint_velocities
            
        #     # If robot is behind path point (velocity points towards path point)
        #     # and distance is significant, maintain current index
        #     if  np.dot(path_vector, ee_velocity) > 0 and np.linalg.norm(path_vector) > 100:
        #         self.mpc.set_path_index(self.path_index)
        #         self.debug_helper.log_state(f"pose is behind, remaining at current index{self.mpc.path_index}")
        #         return



            
        #     # Calculate distance to current target point
        #     dist_to_current = np.hypot(
        #         current_pos[0] - current_path_pos[0],
        #         current_pos[1] - current_path_pos[1]
        #     )
            
        #     # If close enough to current point, move to next point
        #     if dist_to_current <= 10.0:  # Threshold for "close enough"
        #         self.path_index += 1
        #         self.mpc.set_path_index(self.path_index)
        #         self.debug_helper.log_state(f"path index updated to {self.mpc.path_index}")
                
        # except Exception as e:
        #     self.debug_helper.log_state(f"Error in path index update: {str(e)}")

    def _update_tracking_history(self, robot: 'Robot') -> None:
        """
        Update tracking history with current state information.

        Args:
            robot: Current robot state
        """
        try:
            # Store current state
            self.actual_theta_0.append(robot.theta_0)
            self.actual_theta_1.append(robot.theta_1)
            self.actual_omega_0.append(robot.omega_0)
            self.actual_omega_1.append(robot.omega_1)
            
            # Update reference values if available
            if self.path and self.path_index < len(self.path):
                if not self.reference_omega_0:
                    self.reference_omega_0 = []
                    self.reference_omega_1 = []
                if not self.reference_alpha_0:
                    self.reference_alpha_0 = []
                    self.reference_alpha_1 = []
                
                self.reference_omega_0.append(self.path[self.path_index].omega_0)
                self.reference_omega_1.append(self.path[self.path_index].omega_1)
                self.reference_alpha_0.append(self.path[self.path_index].alpha_0)
                self.reference_alpha_1.append(self.path[self.path_index].alpha_1)
            
            # Compute accelerations
            self._compute_accelerations(robot)
            
            # Validate dynamics periodically
            if len(self.actual_theta_0) % 50 == 0:
                self.debug_helper.validate_controller_dynamics(self, robot)
                
        except Exception as e:
            self.debug_helper.log_state(f"Error in tracking history update: {str(e)}")

    def _compute_accelerations(self, robot: 'Robot') -> None:
        """
        Compute and store joint accelerations from velocity history.

        Args:
            robot: Current robot state
        """
        dt = self.constants.DT
        if len(self.actual_omega_0) >= 2:
            alpha_0 = (self.actual_omega_0[-1] - self.actual_omega_0[-2]) / dt
            alpha_1 = (self.actual_omega_1[-1] - self.actual_omega_1[-2]) / dt
            
            max_accel = self.constants.MAX_ACCELERATION * 1.5  # Allow slight overshoot for stability
            
            # Clip accelerations to physical limits
            alpha_0 = np.clip(alpha_0, -max_accel, max_accel)
            alpha_1 = np.clip(alpha_1, -max_accel, max_accel)
            
            self.actual_alpha_0.append(alpha_0)
            self.actual_alpha_1.append(alpha_1)
        else:
            # Initialize with zero acceleration
            self.actual_alpha_0.append(0.0)
            self.actual_alpha_1.append(0.0)
        
        # Update reference accelerations if needed
        if not self.reference_alpha_0:
            self._initialize_reference_accelerations()

    def _initialize_reference_accelerations(self) -> None:
        """
        Initialize reference acceleration profiles from trajectory.
        Uses central differences for interior points and forward/backward
        differences at endpoints.
        """
        dt = self.constants.DT
        path_length = len(self.path)
        self.reference_alpha_0 = [0.0] * path_length
        self.reference_alpha_1 = [0.0] * path_length
        
        # Compute accelerations using central differences
        for i in range(1, path_length - 1):
            self.reference_alpha_0[i] = (self.path[i+1].omega_0 - self.path[i-1].omega_0) / (2 * dt)
            self.reference_alpha_1[i] = (self.path[i+1].omega_1 - self.path[i-1].omega_1) / (2 * dt)
        
        # Handle endpoints using forward/backward differences
        if path_length > 1:
            self.reference_alpha_0[0] = (self.path[1].omega_0 - self.path[0].omega_0) / dt
            self.reference_alpha_1[0] = (self.path[1].omega_1 - self.path[0].omega_1) / dt
            self.reference_alpha_0[-1] = (self.path[-1].omega_0 - self.path[-2].omega_0) / dt
            self.reference_alpha_1[-1] = (self.path[-1].omega_1 - self.path[-2].omega_1) / dt

    def _perform_debug_analysis(self, robot: 'Robot') -> None:
        """
        Perform comprehensive debug analysis of controller performance.

        Args:
            robot: Current robot state

        This method analyzes tracking performance, control signals, and dynamic
        consistency to help identify potential issues during execution.
        """
        self.debug_helper.analyze_tracking_performance(self, robot)
        self.debug_helper.analyze_control_signals(self, robot)
        self.debug_helper.check_dynamic_consistency(self, robot)
        self.debug_helper.print_controller_stats(self, robot)

    def _handle_empty_path(self, robot: 'Robot') -> 'Robot':
        """
        Handle case when no path is available or path is completed.

        Args:
            robot: Current robot state

        Returns:
            Robot: Robot state with zero velocities
        """
        self.debug_helper.log_state("No path available or end of path reached")
        robot.omega_0 = 0.0
        robot.omega_1 = 0.0
        return robot

    def _handle_error(self, robot: 'Robot') -> 'Robot':
        """
        Handle controller error by safely stopping robot.

        Args:
            robot: Current robot state

        Returns:
            Robot: Robot state with zero velocities
        """
        self.debug_helper.log_state("Controller error - stopping robot")
        robot.omega_0 = 0.0
        robot.omega_1 = 0.0
        return robot

    def _reset_controller_states(self) -> None:
        """
        Reset all controller states and tracking variables to initial conditions.
        
        This includes resetting integral terms, previous errors, tracking history,
        and reference trajectories.
        """
        # Reset integral terms and previous errors
        self.integral_error_0 = 0.0
        self.integral_error_1 = 0.0
        self.prev_theta_error_0 = 0.0
        self.prev_theta_error_1 = 0.0
        
        # Reset tracking history
        self.actual_theta_0.clear()
        self.actual_theta_1.clear()
        self.actual_omega_0.clear()
        self.actual_omega_1.clear()
        self.actual_alpha_0.clear()
        self.actual_alpha_1.clear()
        
        # Reset reference trajectories
        self.reference_theta_0.clear()
        self.reference_theta_1.clear()
        self.reference_omega_0.clear()
        self.reference_omega_1.clear()
        self.reference_alpha_0.clear()
        self.reference_alpha_1.clear()
        
        # Reset path tracking
        self.path_index = 0
        
        self.debug_helper.log_state("Controller states reset")

    def is_planning_timeout(self) -> bool:
        """
        Check if path planning has exceeded timeout duration.

        Returns:
            bool: True if planning has timed out, False otherwise
        """
        if self.planning_mode and self.plan_start_time:
            timeout_occurred = (time.time() - self.plan_start_time) > self.planning_timeout
            if timeout_occurred:
                self.debug_helper.log_state("Planning Timeout!")
            return timeout_occurred
        return False