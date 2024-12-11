from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import heapq
from planner.grid_planner import GridBasedPlanner
from enviroment.robot import Robot
from enviroment.world import World
from utils.state import State
from utils.debug_helper import DebugHelper


class AStarPlanner(GridBasedPlanner):
    """
    An optimized A* motion planner for 2-link robotic manipulators.
    
    Implements both kinodynamic and coarse planning modes with trajectory optimization.
    Kinodynamic planning considers dynamic constraints like velocities and accelerations,
    while coarse planning focuses on geometric path finding.

    Attributes:
        mode (str): Planning mode ('kinodynamic' or 'coarse')
        velocity_resolution (float): Minimum velocity change increment
        max_smoothing_error (float): Maximum allowed deviation during path smoothing
        motion_primitives (List[Tuple[float, float]]): Set of possible motion actions
        smoothness_params (Dict): Parameters controlling path smoothness
    """

    def __init__(self, robot: Robot, world: World, mode: str = 'kinodynamic'):
        """
        Initialize the A* planner with robot model and world representation.

        Args:
            robot (Robot): Robot model containing kinematics and physical parameters
            world (World): World model containing obstacles and boundaries
            mode (str): Planning mode ('kinodynamic' or 'coarse')
        """
        super().__init__(robot, world)
        self.mode = mode
        self.velocity_resolution = self.constants.VELOCITY_RESOLUTION
        self.max_smoothing_error = 0.1
        self.min_path_points = 5
        self.motion_primitives = self._generate_motion_primitives() if mode == 'kinodynamic' else None
        self.last_progress_time = 0

        # Smoothing and optimization parameters
        self.max_jerk = self.constants.MAX_JERK
        self.smoothing_window = 5
        self.min_velocity_threshold = 0.1
        self.acceleration_weight = 0.3
        self.jerk_weight = 0.2
        
        self.debug_helper = DebugHelper(debug=False)

    def set_mode(self, mode: str) -> None:
        """
        Switch between planning modes and update motion primitives accordingly.

        Args:
            mode (str): Target planning mode ('kinodynamic' or 'coarse')
        
        Raises:
            ValueError: If mode is not 'kinodynamic' or 'coarse'
        """
        if mode not in ['kinodynamic', 'coarse']:
            raise ValueError("Mode must be either 'kinodynamic' or 'coarse'")
        self.mode = mode
        self.motion_primitives = self._generate_motion_primitives() if mode == 'kinodynamic' else None

    def get_neighbors_coarse(self, current_state: State) -> List[State]:
        """
        Generate neighboring states for coarse planning with adaptive resolution.
        Uses Jacobian-based sampling for better goal convergence.

        Args:
            current_state (State): Current robot state

        Returns:
            List[State]: List of valid neighboring states
        """
        neighbors = []
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(self.goal_state.theta_0, self.goal_state.theta_1)
        dist_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
        
        # Adaptive resolution based on distance to goal
        base_resolution = self.constants.THETA_0_RESOLUTION
        theta_resolution = (base_resolution * 0.5 if dist_to_goal < 1.0 else 
                          base_resolution * 2.0 if dist_to_goal < 5.0 else 
                          base_resolution * 3.0)
            
        angle_steps = [
            -1.5 * theta_resolution, -0.25 * theta_resolution, 
            0.25 * theta_resolution, 1.5 * theta_resolution,
        ]

        # Add goal-biased samples using Jacobian
        goal_direction = np.array([goal_pos[0] - current_pos[0], goal_pos[1] - current_pos[1]])
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
            jacobian = self._compute_jacobian(current_state.theta_0, current_state.theta_1)
            
            if jacobian is not None:
                try:
                    lambda_factor = 0.1
                    J_T = jacobian.T
                    inv_term = np.linalg.inv(jacobian @ J_T + lambda_factor * np.eye(2))
                    joint_direction = J_T @ inv_term @ goal_direction
                    
                    bias_scales = [0.5, 1.5, 2.5, 3.5]
                    angle_steps.extend([joint_direction[0] * scale * theta_resolution 
                                     for scale in bias_scales])
                    angle_steps.extend([joint_direction[1] * scale * theta_resolution 
                                     for scale in bias_scales])
                except np.linalg.LinAlgError:
                    pass

        # Generate and validate neighbors
        for d_theta0 in angle_steps:
            for d_theta1 in angle_steps:
                theta_0_new = current_state.theta_0 + d_theta0
                theta_1_new = current_state.theta_1 + d_theta1

                if abs(d_theta0) < 1e-6 and abs(d_theta1) < 1e-6:
                    continue

                if not self.within_joint_limits((theta_0_new, theta_1_new)):
                    continue

                neighbor = State(theta_0_new, theta_1_new, 0.0, 0.0)
                if not self.is_collision(neighbor):
                    neighbor_pos = self.robot.forward_kinematics(theta_0_new, theta_1_new)
                    if (self._is_in_workspace(neighbor_pos) and 
                        self._check_step_feasibility(current_pos, neighbor_pos)):
                        neighbors.append(neighbor)

        return neighbors

    def _compute_jacobian(self, theta_0: float, theta_1: float) -> Optional[np.ndarray]:
        """
        Compute the manipulator Jacobian matrix at given joint angles.

        Args:
            theta_0 (float): First joint angle
            theta_1 (float): Second joint angle

        Returns:
            Optional[np.ndarray]: 2x2 Jacobian matrix or None if computation fails
        """
        l1, l2 = self.robot.constants.LINK_1, self.robot.constants.LINK_2
        c1, s1 = np.cos(theta_0), np.sin(theta_0)
        c12, s12 = np.cos(theta_0 + theta_1), np.sin(theta_0 + theta_1)
        
        return np.array([
            [-l1*s1 - l2*s12, -l2*s12],
            [l1*c1 + l2*c12, l2*c12]
        ])

    def _check_step_feasibility(self, current_pos: Tuple[float, float], 
                              next_pos: Tuple[float, float], 
                              max_step: float = 0.5) -> bool:
        """
        Verify if the step between two positions respects maximum step size constraints.

        Args:
            current_pos (Tuple[float, float]): Current end-effector position
            next_pos (Tuple[float, float]): Proposed next position
            max_step (float): Maximum allowed step size in workspace

        Returns:
            bool: True if step is feasible, False otherwise
        """
        step_size = np.hypot(next_pos[0] - current_pos[0], 
                           next_pos[1] - current_pos[1])
        return step_size <= max_step

    def _is_in_workspace(self, pos: Tuple[float, float]) -> bool:
        """
        Check if a position lies within the robot's reachable workspace.

        Args:
            pos (Tuple[float, float]): Position to check

        Returns:
            bool: True if position is reachable, False otherwise
        """
        dist_from_base = np.hypot(pos[0], pos[1])
        min_reach = self.robot.constants.min_reachable_radius()
        max_reach = self.robot.constants.max_reachable_radius()
        return min_reach <= dist_from_base <= max_reach

    def heuristic_coarse(self, current: State, goal: State) -> float:
        """
        Calculate heuristic estimate for coarse planning mode.
        Uses end-effector distance in workspace.

        Args:
            current (State): Current robot state
            goal (State): Goal robot state

        Returns:
            float: Estimated cost to goal
        """
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        return np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])

    def is_goal_coarse(self, current_state: State, goal_state: State) -> bool:
        """
        Check if current state satisfies goal conditions in coarse planning mode.
        Uses adaptive thresholds based on workspace size.

        Args:
            current_state (State): Current robot state
            goal_state (State): Goal robot state

        Returns:
            bool: True if goal conditions are met, False otherwise
        """
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(goal_state.theta_0, goal_state.theta_1)
        distance_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
        
        # Adaptive thresholds based on robot dimensions
        l1, l2 = self.robot.constants.LINK_1, self.robot.constants.LINK_2
        workspace_size = l1 + l2
        position_threshold = min(0.2, workspace_size * 0.05)
        
        return distance_to_goal <= position_threshold

    def _generate_motion_primitives(self) -> List[Tuple[float, float]]:
        """
        Generate set of motion primitives for kinodynamic planning.
        Uses conservative acceleration levels for smooth motion.

        Returns:
            List[Tuple[float, float]]: List of acceleration pairs for joints
        """
        max_alpha = self.constants.MAX_ACCELERATION
        coarse_levels = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
        acceleration_levels = [max_alpha * factor for factor in coarse_levels]
        
        return [(alpha_0, alpha_1) 
                for alpha_0 in acceleration_levels 
                for alpha_1 in acceleration_levels]

    def get_neighbors(self, current_state: State) -> List[State]:
        """
        Generate neighboring states for kinodynamic planning.
        Implements velocity-based state expansion with deceleration profile.

        Args:
            current_state (State): Current robot state

        Returns:
            List[State]: List of dynamically feasible neighboring states
        """
        neighbors = []
        dt = self.constants.DT
        max_vel = self.constants.MAX_VELOCITY
        min_vel = self.constants.MIN_VELOCITY
        resolution = self.constants.THETA_0_RESOLUTION

        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(self.goal_state.theta_0, self.goal_state.theta_1)
        dist_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])

        # Apply deceleration profile near goal
        decel_start_dist = 2.0
        if dist_to_goal < decel_start_dist:
            vel_scale = min(1.0, dist_to_goal / decel_start_dist)
            max_vel *= vel_scale
            min_vel *= vel_scale
        
        for alpha_0, alpha_1 in self.motion_primitives:
            # Scale accelerations near goal
            if dist_to_goal < decel_start_dist:
                alpha_0 *= vel_scale
                alpha_1 *= vel_scale

            # Velocity update with limits
            omega_0_new = np.clip(
                current_state.omega_0 + alpha_0 * dt,
                -max_vel, max_vel
            )
            omega_1_new = np.clip(
                current_state.omega_1 + alpha_1 * dt,
                -max_vel, max_vel
            )

            # Enforce minimum velocity when moving
            if abs(current_state.theta_0 - self.goal_state.theta_0) > 0.01:
                omega_0_new = np.clip(abs(omega_0_new), min_vel, max_vel) * np.sign(omega_0_new)
            if abs(current_state.theta_1 - self.goal_state.theta_1) > 0.01:
                omega_1_new = np.clip(abs(omega_1_new), min_vel, max_vel) * np.sign(omega_1_new)
            
            # Position update using RK4 integration
            k1_0, k1_1 = current_state.omega_0, current_state.omega_1
            k2_0, k2_1 = omega_0_new, omega_1_new
            
            theta_0_new = current_state.theta_0 + dt * (k1_0 + k2_0) / 2
            theta_1_new = current_state.theta_1 + dt * (k1_1 + k2_1) / 2
            
            # Discretize and validate new state
            theta_0_new = round(theta_0_new / resolution) * resolution
            theta_1_new = round(theta_1_new / resolution) * resolution
            
            if self.within_joint_limits((theta_0_new, theta_1_new)):
                neighbor = State(theta_0_new, theta_1_new, omega_0_new, omega_1_new)
                if self._check_kinematic_feasibility(current_state, neighbor):
                    neighbors.append(neighbor)
        
        return neighbors

    def _check_kinematic_feasibility(self, current: State, next: State) -> bool:
        """
        Verify kinematic feasibility of state transition with relaxed constraints.

        Args:
            current (State): Current robot state
            next (State): Proposed next state

        Returns:
            bool: True if transition is feasible, False otherwise
        """
        dt = self.constants.DT
        max_accel = self.constants.MAX_ACCELERATION
        
        alpha_0 = (next.omega_0 - current.omega_0) / dt
        alpha_1 = (next.omega_1 - current.omega_1) / dt
        
        accel_tolerance = 1.5
        max_allowed = max_accel * accel_tolerance
        
        if abs(alpha_0) > max_allowed or abs(alpha_1) > max_allowed:
            return False
            
        max_vel_change = self.constants.MAX_VELOCITY * 0.5
        if (abs(next.omega_0 - current.omega_0) > max_vel_change or 
            abs(next.omega_1 - current.omega_1) > max_vel_change):
            return False
        
        return True

    def Plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot: Robot, 
             final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """
        Plan a path from start to goal with automatic mode selection and trajectory optimization.
        Attempts coarse planning first, then refines with kinodynamic constraints if needed.

        Args:
            start (Tuple[float, float]): Start position in workspace
            goal (Tuple[float, float]): Goal position in workspace
            robot (Robot): Robot model
            final_velocities (Tuple[float, float]): Desired final joint velocities

        Returns:
            List[State]: Planned trajectory or empty list if planning fails
        """
        try:
            # First attempt coarse planning
            self.set_mode('coarse')
            coarse_path = self.plan(start, goal, self.robot)

            if not coarse_path:
                self.debug_helper.log_state("Coarse planning failed, attempting kinodynamic planning")
                self.set_mode('kinodynamic')
                path = self.plan(start, goal, self.robot, final_velocities)
                return path if self._validate_path(path) else []
                
            # Refine coarse path if close enough to goal
            self.set_mode('kinodynamic')            
            if self.best_distance_to_goal < 2.0:
                return coarse_path if self._validate_path(coarse_path) else []
            
            # Attempt connecting coarse path segments to goal
            return self._connect_coarse_to_goal(coarse_path, goal, final_velocities)
                        
        except Exception as e:
            self.debug_helper.log_state(f"Planning error: {str(e)}")
            return []

    def planbid(self, start: Tuple[float, float], goal: Tuple[float, float], robot: Robot, 
                final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """
        Plan a path using bidirectional A* search.
        Simultaneously explores from both start and goal states.

        Args:
            start (Tuple[float, float]): Start position in workspace
            goal (Tuple[float, float]): Goal position in workspace
            robot (Robot): Robot model
            final_velocities (Tuple[float, float]): Desired final velocities

        Returns:
            List[State]: Planned trajectory or empty list if planning fails

        Raises:
            ValueError: If no path can be found
        """
        # Initialize planning state
        self.explored_states_count = 0
        self.best_distance_to_goal = float('inf')
        start_time = time.time()

        # Set up forward search
        forward_start_state = State(robot.theta_0, robot.theta_1, robot.omega_0, robot.omega_1)
        goal_theta = robot.inverse_kinematics(*goal)
        forward_goal_state = State(goal_theta[0], goal_theta[1], final_velocities[0], final_velocities[1])
        self.goal_state = forward_goal_state

        # Set up backward search
        backward_start_state = forward_goal_state
        backward_goal_state = forward_start_state

        # Initialize search queues and tracking structures
        forward_open = [(self.heuristic(forward_start_state, forward_goal_state), 0, forward_start_state)]
        backward_open = [(self.heuristic(backward_start_state, backward_goal_state), 0, backward_start_state)]
        
        forward_closed: Set[State] = set()
        backward_closed: Set[State] = set()
        
        forward_came_from: Dict[State, Optional[State]] = {forward_start_state: None}
        backward_came_from: Dict[State, Optional[State]] = {backward_start_state: None}
        
        forward_g_score: Dict[State, float] = {forward_start_state: 0}
        backward_g_score: Dict[State, float] = {backward_start_state: 0}

        best_path = None
        best_distance = float('inf')
        meeting_point = None

        while forward_open and backward_open:
            # Forward search expansion
            if forward_open:
                current_forward = forward_open[0][2]
                current_forward_pos = robot.forward_kinematics(current_forward.theta_0, current_forward.theta_1)
                forward_distance = np.hypot(current_forward_pos[0] - goal[0], 
                                          current_forward_pos[1] - goal[1])

                # Check for meeting point between searches
                for state in backward_closed:
                    if self.states_close_enough(current_forward, state):
                        meeting_point = (current_forward, state)
                        break

                # Update best path if improved
                if forward_distance < best_distance:
                    best_distance = forward_distance
                    best_path = self.reconstruct_bidirectional_path(
                        current_forward, forward_came_from, backward_came_from, meeting_point)
                    self.last_progress_time = time.time()

                _, _, current = heapq.heappop(forward_open)
                forward_closed.add(current)

                # Expand forward neighbors
                self._expand_neighbors(
                    current, forward_closed, forward_came_from, forward_g_score,
                    forward_open, forward_goal_state
                )

            # Backward search expansion
            if backward_open:
                current_backward = backward_open[0][2]
                
                # Check for meeting point
                for state in forward_closed:
                    if self.states_close_enough(current_backward, state):
                        meeting_point = (state, current_backward)
                        break

                _, _, current = heapq.heappop(backward_open)
                backward_closed.add(current)

                # Expand backward neighbors
                self._expand_neighbors(
                    current, backward_closed, backward_came_from, backward_g_score,
                    backward_open, backward_goal_state
                )

            # Check termination conditions
            if self._check_termination_conditions(start_time, best_path):
                if meeting_point:
                    return self.reconstruct_bidirectional_path(
                        meeting_point[0], forward_came_from, backward_came_from, meeting_point)
                return best_path if best_path else []

            # Memory management and state tracking
            self._manage_planning_resources()

        raise ValueError("No path found")

    def _expand_neighbors(self, current: State, closed_set: Set[State], 
                        came_from: Dict[State, Optional[State]], 
                        g_score: Dict[State, float],
                        open_queue: List[Tuple[float, int, State]],
                        goal_state: State) -> None:
        """
        Expand neighboring states and update planning structures.

        Args:
            current (State): Current state to expand from
            closed_set (Set[State]): Set of explored states
            came_from (Dict[State, Optional[State]]): Parent mapping
            g_score (Dict[State, float]): Cost-to-reach mapping
            open_queue (List[Tuple[float, int, State]]): Priority queue of states to explore
            goal_state (State): Goal state for heuristic calculation
        """
        for neighbor in self.get_neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + self.distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                if not self.is_collision(neighbor):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal_state)
                    heapq.heappush(open_queue, (f_score, self.explored_states_count, neighbor))

    def _check_termination_conditions(self, start_time: float, best_path: Optional[List[State]]) -> bool:
        """
        Check if planning should terminate based on time or progress.

        Args:
            start_time (float): Time when planning started
            best_path (Optional[List[State]]): Best path found so far

        Returns:
            bool: True if planning should terminate, False otherwise
        """
        current_time = time.time()
        
        # Check overall timeout
        if current_time - start_time > self.planning_timeout:
            self.debug_helper.log_state("Planning timeout - returning best path")
            return True

        # Check progress timeout
        if current_time - self.last_progress_time > 5.0 and best_path:
            self.debug_helper.log_state("No progress - returning best path")
            return True

        return False

    def _manage_planning_resources(self) -> None:
        """
        Manage memory and state exploration resources during planning.
        Performs cache cleanup and tracks explored states.
        """
        # Periodic cache cleanup
        if self.explored_states_count % 1000 == 0:
            self.clear_cache()

        self.explored_states_count += 1
        if self.explored_states_count > self.max_explored_states:
            raise ValueError("Maximum number of states explored")

    def states_close_enough(self, state1: State, state2: State, threshold: float = 0.1) -> bool:
        """
        Check if two states can be kinodynamically connected.

        Args:
            state1 (State): First state
            state2 (State): Second state
            threshold (float): Maximum allowed position difference

        Returns:
            bool: True if states can be connected, False otherwise
        """
        # Check end-effector position difference
        pos1 = self.robot.forward_kinematics(state1.theta_0, state1.theta_1)
        pos2 = self.robot.forward_kinematics(state2.theta_0, state2.theta_1)
        pos_diff = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        
        # Check velocity compatibility
        vel_diff = np.hypot(state1.omega_0 - state2.omega_0, 
                          state1.omega_1 - state2.omega_1)
        
        # Verify acceleration feasibility
        dt = self.constants.DT
        required_accel_0 = abs(state2.omega_0 - state1.omega_0) / dt
        required_accel_1 = abs(state2.omega_1 - state1.omega_1) / dt
        
        accel_feasible = (required_accel_0 <= self.constants.MAX_ACCELERATION and 
                         required_accel_1 <= self.constants.MAX_ACCELERATION)
        
        return (pos_diff < threshold and 
                vel_diff < self.constants.VELOCITY_RESOLUTION and 
                accel_feasible)

    def interpolate_states(self, state1: State, state2: State, num_steps: int = 10) -> List[State]:
        """
        Generate kinodynamically feasible interpolation between states.

        Args:
            state1 (State): Starting state
            state2 (State): Ending state
            num_steps (int): Number of interpolation steps

        Returns:
            List[State]: List of interpolated states
        """
        states = []
        dt = self.constants.DT
        
        # Calculate required accelerations
        delta_omega_0 = state2.omega_0 - state1.omega_0
        delta_omega_1 = state2.omega_1 - state1.omega_1
        
        alpha_0 = delta_omega_0 / (num_steps * dt)
        alpha_1 = delta_omega_1 / (num_steps * dt)
        
        # Generate intermediate states with constant acceleration
        for i in range(num_steps + 1):
            t = i * dt
            
            omega_0 = state1.omega_0 + alpha_0 * t
            omega_1 = state1.omega_1 + alpha_1 * t
            
            theta_0 = state1.theta_0 + state1.omega_0 * t + 0.5 * alpha_0 * t * t
            theta_1 = state1.theta_1 + state1.omega_1 * t + 0.5 * alpha_1 * t * t
            
            states.append(State(theta_0, theta_1, omega_0, omega_1))
        
        return states

    def _validate_path(self, path: List[State]) -> bool:
        """
        Validate a planned path against kinematic and dynamic constraints.

        Args:
            path (List[State]): Path to validate

        Returns:
            bool: True if path is valid, False otherwise
        """
        if not path:
            return False
            
        self.debug_helper.print_path_stats(path, self.robot)
        self.debug_helper.validate_path_limits(path, self.robot)
        self.debug_helper.print_path_points(path)
        return True

    def _connect_coarse_to_goal(self, coarse_path: List[State], goal: Tuple[float, float], 
                               final_velocities: Tuple[float, float]) -> List[State]:
        """
        Attempt to connect coarse path segments to goal using kinodynamic planning.

        Args:
            coarse_path (List[State]): Initial coarse path
            goal (Tuple[float, float]): Goal position
            final_velocities (Tuple[float, float]): Desired final velocities

        Returns:
            List[State]: Connected path or empty list if connection fails
        """
        path_segments = 5
        for i in range(path_segments, 0, -1):
            intermediate_state = coarse_path[(int(len(coarse_path) * i / path_segments)) - 1]
            new_goal = (intermediate_state.theta_0, intermediate_state.theta_1)
            velocities = (getattr(intermediate_state, 'velocity_0', 0.0),
                         getattr(intermediate_state, 'velocity_1', 0.0))
            
            path = self.plan(goal, new_goal, self.robot, velocities)
            if path:
                path.reverse()
                split_index = (int(len(coarse_path) * i / path_segments)) - 1
                final_path = coarse_path[:split_index]
                final_path.extend(path)
                return final_path if self._validate_path(final_path) else []
                
        return []