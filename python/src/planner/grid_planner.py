from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from planner.base_planner import Planner
from enviroment.robot import Robot
from enviroment.world import World
from enviroment.obstacles import Obstacle
from utils.state import State
from utils.debug_helper import DebugHelper
from enviroment.collision_checker import CollisionChecker

class GridBasedPlanner(Planner):
    """
    A grid-based motion planner for robotic manipulators that uses discretized configuration space.
    
    This planner implements path planning considering both joint space and workspace constraints,
    with adaptive resolution based on obstacle proximity.
    """

    def __init__(self, robot: Robot, world: World):
        """
        Initialize the grid-based planner with robot and world models.

        Args:
            robot (Robot): Robot model containing kinematics and physical parameters
            world (World): World model containing obstacles and boundaries
        """
        self.robot = robot
        self.world = world
        self.constants = robot.constants
        self.resolutions = (self.constants.THETA_0_RESOLUTION, self.constants.THETA_1_RESOLUTION)
        self.collision_cache: Dict[State, bool] = {}
        self.max_cache_size = 10000
        self.planning_timeout = 300.0  # Maximum planning time in seconds
        self.progress_timeout = 150.0  # Maximum time without progress
        self.min_progress_threshold = 0.1
        self.max_explored_states = 10000

    def clear_cache(self):
        """
        Clear the collision checking cache when it exceeds the maximum size.
        Maintains the most recent entries up to max_cache_size.
        """
        if len(self.collision_cache) > self.max_cache_size:
            items = list(self.collision_cache.items())
            self.collision_cache = dict(items[-self.max_cache_size:])

    def heuristic(self, current: State, goal: State) -> float:
        """
        Calculate heuristic cost between current and goal states using a weighted combination
        of workspace distance, configuration space distance, and velocity alignment.

        Args:
            current (State): Current robot state
            goal (State): Goal robot state

        Returns:
            float: Estimated cost to goal
        """
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        
        workspace_distance = np.hypot(current_pos[0] - goal_pos[0], 
                                    current_pos[1] - goal_pos[1])
        config_distance = np.hypot(current.theta_0 - goal.theta_0, 
                                 current.theta_1 - goal.theta_1)
        velocity_alignment = np.hypot(current.omega_0 - goal.omega_0,
                                    current.omega_1 - goal.omega_1)
        
        # Weighted combination of metrics
        w1, w2, w3 = 1.0, 0.3, 0.1
        return w1 * workspace_distance + w2 * config_distance + w3 * velocity_alignment

    def distance(self, from_state: State, to_state: State) -> float:
        """
        Calculate actual cost between two adjacent states.

        Args:
            from_state (State): Starting state
            to_state (State): Ending state

        Returns:
            float: Actual cost between states
        """
        delta_theta = np.hypot(to_state.theta_0 - from_state.theta_0, 
                             to_state.theta_1 - from_state.theta_1)
        delta_omega = np.hypot(to_state.omega_0 - from_state.omega_0, 
                             to_state.omega_1 - from_state.omega_1)
        return delta_theta + delta_omega

    def within_joint_limits(self, node: Tuple[float, float]) -> bool:
        """
        Check if joint angles are within the robot's joint limits.

        Args:
            node (Tuple[float, float]): Joint angles to check

        Returns:
            bool: True if within limits, False otherwise
        """
        theta_0, theta_1 = node
        return (self.constants.JOINT_LIMITS[0] <= theta_0 <= self.constants.JOINT_LIMITS[1] and
                self.constants.JOINT_LIMITS[0] <= theta_1 <= self.constants.JOINT_LIMITS[1])

    def is_collision(self, state: State) -> bool:
        """
        Check if a state results in collision with obstacles or self-collision.
        Uses caching to improve performance.

        Args:
            state (State): Robot state to check

        Returns:
            bool: True if in collision, False otherwise
        """
        if state in self.collision_cache:
            return self.collision_cache[state]

        # Check both obstacle and self-collision
        in_collision = self.check_obstacle_collision(state.theta_0, state.theta_1) #or 
                  #     self.robot.self_collision(state.theta_0, state.theta_1))
        
        self.collision_cache[state] = in_collision
        return in_collision

    def is_goal(self, current_state: State, goal_state: State) -> bool:
        """
        Check if current state satisfies goal conditions based on end-effector position
        and joint velocities.

        Args:
            current_state (State): Current robot state
            goal_state (State): Goal robot state

        Returns:
            bool: True if goal conditions are met, False otherwise
        """
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(goal_state.theta_0, goal_state.theta_1)
        
        position_threshold = 0.5
        velocity_threshold = self.constants.VELOCITY_RESOLUTION
        
        position_close = np.hypot(current_pos[0] - goal_pos[0], 
                                current_pos[1] - goal_pos[1]) <= position_threshold
        
        return position_close

    def check_obstacle_collision(self, theta_0: float, theta_1: float) -> bool:
        """
        Check for collisions between robot links and obstacles.

        Args:
            theta_0 (float): First joint angle
            theta_1 (float): Second joint angle

        Returns:
            bool: True if collision detected, False otherwise
        """
        link_segments = self.robot.compute_link_segments(theta_0, theta_1)
        return any(self.robot_collision(segment, obstacle) 
                  for obstacle in self.world.inflated_obstacles 
                  for segment in link_segments)

    def robot_collision(self, segment: Tuple[Tuple[float, float], Tuple[float, float]], 
                       obstacle: Obstacle) -> bool:
        """
        Check collision between a robot link segment and an obstacle.

        Args:
            segment (Tuple[Tuple[float, float], Tuple[float, float]]): Link segment endpoints
            obstacle (Obstacle): Obstacle to check against

        Returns:
            bool: True if collision detected, False otherwise
        """
        p1, p2 = segment
        if obstacle.shape == 'circle':
            return CollisionChecker.line_circle_collision(p1, p2, obstacle.position, obstacle.size)
        elif obstacle.shape == 'rectangle':
            return CollisionChecker.line_rectangle_collision(p1, p2, obstacle.position, obstacle.size)
        return False

    def reconstruct_path(self, came_from: Dict[State, Optional[State]], 
                        current: State) -> List[State]:
        """
        Reconstruct the path from start to goal using the came_from mapping.

        Args:
            came_from (Dict[State, Optional[State]]): Parent mapping for each state
            current (State): Goal state to reconstruct path from

        Returns:
            List[State]: Path from start to goal
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            if current is None:
                break
            total_path.append(current)
        return total_path[::-1]

    def check_timeout_and_progress(self, current: State, goal: State, start_time: float):
        """
        Monitor planning progress and check for timeout conditions.

        Args:
            current (State): Current planning state
            goal (State): Goal state
            start_time (float): Time when planning started

        Raises:
            PlanningTimeoutError: If planning timeout or progress timeout is reached
        """
        current_time = time.time()
        if current_time - start_time > self.planning_timeout:
            raise PlanningTimeoutError("Planning timeout reached")

        current_distance = self.heuristic(current, goal)
        if current_distance < self.best_distance_to_goal:
            self.best_distance_to_goal = current_distance
            self.last_progress_time = current_time
        elif current_time - self.last_progress_time > self.progress_timeout:
            raise PlanningTimeoutError("No progress made in planning")


class PlanningTimeoutError(Exception):
    """Exception raised when planning exceeds time limits or makes no progress."""
    pass