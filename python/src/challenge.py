import time
from typing import List, Tuple, Union, Dict, Set, Optional
import numpy as np
import pygame
import json
import heapq
import itertools

from collision_checker import CollisionChecker
from utils import generate_random_goal, generate_random_rectangle_obstacle
from obstacle import Obstacle
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    theta_0: float
    theta_1: float
    omega_0: float
    omega_1: float

class RobotConstants:
    def __init__(self, config):
        self.BASE_RADIUS = config.get('base_radius', 10.0)
        self.JOINT_LIMITS = tuple(config['joint_limits'])
        self.MAX_VELOCITY = config['max_velocity']
        self.MAX_ACCELERATION = config['max_acceleration']
        self.VELOCITY_RESOLUTION = config['velocity_resolution']
        self.DT = config['dt']
        self.LINK_1 = config['link_lengths'][0]
        self.LINK_2 = config['link_lengths'][1]
        self.ROBOT_ORIGIN = tuple(config.get('robot_origin', (0.0, 0.0)))

        # Discretization parameters for planners
        self.THETA_0_RESOLUTION = config.get('theta_0_resolution', 0.1)
        self.THETA_1_RESOLUTION = config.get('theta_1_resolution', 0.1)

    def min_reachable_radius(self) -> float:
        return max(self.LINK_1 - self.LINK_2, 0)

    def max_reachable_radius(self) -> float:
        return self.LINK_1 + self.LINK_2


class Robot:
    """
    Represents a two-link robotic arm with constraints on joint angles, velocity, and acceleration.
    """

    def __init__(self, constants: RobotConstants) -> None:
        self.constants = constants
        # Initialize angles and their histories
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self._theta_0 = 0
        self._theta_1 = 1.57
        self.omega_0 = 0.0
        self.omega_1 = 0.0

    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self.all_theta_0.append(value)
        self._theta_0 = value
        self._validate_joint_limits(value, 0)
        # self._validate_velocity(self.all_theta_0, 0)
        # self._validate_acceleration(self.all_theta_0, 0)

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self.all_theta_1.append(value)
        self._theta_1 = value
        self._validate_joint_limits(value, 1)
        # self._validate_velocity(self.all_theta_1, 1)
        # self._validate_acceleration(self.all_theta_1, 1)

    def _validate_joint_limits(self, theta: float, joint_id: int) -> None:
        if not (self.constants.JOINT_LIMITS[0] <= theta <= self.constants.JOINT_LIMITS[1]):
            raise ValueError(f"Joint {joint_id} angle {theta} exceeds joint limits.")

    def _validate_velocity(self, all_theta: List[float], joint_id: int) -> None:
        velocity = self._max_velocity(all_theta)
        if velocity > self.constants.MAX_VELOCITY:
            raise ValueError(f"Joint {joint_id} velocity {velocity} exceeds limit.")

    def _validate_acceleration(self, all_theta: List[float], joint_id: int) -> None:
        acceleration = self._max_acceleration(all_theta)
        if acceleration > self.constants.MAX_ACCELERATION:
            raise ValueError(f"Joint {joint_id} acceleration {acceleration} exceeds limit.")

    def joint_1_pos(self, theta_0: Optional[float] = None) -> Tuple[float, float]:
        """Compute the position of the first joint."""
        if theta_0 is None:
            theta_0 = self.theta_0
        return (
            self.constants.LINK_1 * np.cos(theta_0),
            self.constants.LINK_1 * np.sin(theta_0),
        )

    def joint_2_pos(self, theta_0: Optional[float] = None, theta_1: Optional[float] = None) -> Tuple[float, float]:
        """Compute the position of the end-effector."""
        if theta_0 is None:
            theta_0 = self.theta_0
        if theta_1 is None:
            theta_1 = self.theta_1
        return self.forward_kinematics(theta_0, theta_1)

    def compute_link_segments(self, theta_0: float, theta_1: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Compute line segments representing the robot's links."""
        j0 = self.constants.ROBOT_ORIGIN
        j1 = self.joint_1_pos(theta_0)
        j2 = self.forward_kinematics(theta_0, theta_1)
        return [(j0, j1), (j1, j2)]

    def forward_kinematics(self, theta_0: float, theta_1: float) -> Tuple[float, float]:
        """Compute the end-effector position given joint angles."""
        x = (
            self.constants.LINK_1 * np.cos(theta_0)
            + self.constants.LINK_2 * np.cos(theta_0 + theta_1)
        )
        y = (
            self.constants.LINK_1 * np.sin(theta_0)
            + self.constants.LINK_2 * np.sin(theta_0 + theta_1)
        )
        return x, y

    def inverse_kinematics(self, x: float, y: float) -> Tuple[float, float]:
        """Compute joint angles from the position of the end-effector."""
        cos_theta_1 = (
            (x**2 + y**2 - self.constants.LINK_1**2 - self.constants.LINK_2**2)
            / (2 * self.constants.LINK_1 * self.constants.LINK_2)
        )
        cos_theta_1 = np.clip(cos_theta_1, -1.0, 1.0)
        theta_1 = np.arccos(cos_theta_1)
        k1 = self.constants.LINK_1 + self.constants.LINK_2 * np.cos(theta_1)
        k2 = self.constants.LINK_2 * np.sin(theta_1)
        theta_0 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return theta_0, theta_1

    def _max_velocity(self, all_theta: List[float]) -> float:
        """Calculate the maximum velocity from joint angle history."""
        diffs = np.diff(all_theta)
        if len(diffs) == 0:
            return 0.0
        return float(max(abs(diffs)) / self.constants.DT)

    def _max_acceleration(self, all_theta: List[float]) -> float:
        """Calculate the maximum acceleration from joint angle history."""
        diffs = np.diff(all_theta)
        if len(diffs) < 2:
            return 0.0
        return float(max(abs(np.diff(diffs))) / self.constants.DT**2)

    def self_collision(self, theta_0: float, theta_1: float) -> bool:
        # Positions of the joints
        base = self.constants.ROBOT_ORIGIN
        joint1 = self.joint_1_pos(theta_0)
        joint2 = self.joint_2_pos(theta_0, theta_1)

        # Check if link2 collides with the base circle
        if CollisionChecker.line_circle_collision(joint1, joint2, base, self.constants.BASE_RADIUS):
            return True

        return False


class Obstacle:
    def __init__(self, shape: str, position: Tuple[float, float], size: Union[float, Tuple[float, float]]):
        self.shape = shape  # 'circle' or 'rectangle'
        self.position = position
        self.size = size


class World:
    """
    Represents the environment where the robot operates.
    """

    def __init__(self, width: int, height: int, robot_origin: Tuple[int, int], obstacles=None) -> None:
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.obstacles = obstacles if obstacles is not None else []

    def convert_to_display(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """
        Converts robot coordinates to display coordinates for rendering.
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin
        return int(offset_x + robot_x), int(offset_y - robot_y)


class Visualizer:
    """
    Handles rendering of the robot and the world using pygame.
    """

    def __init__(self, world: World, config: dict) -> None:
        """
        Initializes the pygame environment and rendering settings.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption("Robot Simulation")
        self.font = pygame.font.SysFont(None, 30)
        self.colors = {
            'background': tuple(config['visualizer']['colors']['background']),
            'robot': tuple(config['visualizer']['colors']['robot']),
            'goal': tuple(config['visualizer']['colors']['goal']),
            'obstacle': tuple(config['visualizer']['colors']['obstacle']),
            'success_text': tuple(config['visualizer']['colors']['success_text']),
            'path': tuple(config['visualizer']['colors'].get('path', [0, 0, 255]))
        }

    def display_world(self, goal: Tuple[float, float]) -> None:
        """
        Renders the goal position and obstacles in the world.
        """
        self.display_goal(goal)
        self.display_obstacles()

    def display_goal(self, goal: Tuple[float, float]) -> None:
        goal = self.world.convert_to_display(goal)
        pygame.draw.circle(self.screen, self.colors['goal'], goal, 6)

    def display_obstacles(self) -> None:
        for obstacle in self.world.obstacles:
            if obstacle.shape == 'circle':
                position = self.world.convert_to_display(obstacle.position)
                pygame.draw.circle(self.screen, self.colors['obstacle'], position, int(obstacle.size))
            elif obstacle.shape == 'rectangle':
                width, height = obstacle.size
                # Compute display coordinates for the top-left corner
                left = self.world.robot_origin[0] + obstacle.position[0]
                top = self.world.robot_origin[1] - (obstacle.position[1] + height)
                rect = pygame.Rect(left, top, width, height)
                pygame.draw.rect(self.screen, self.colors['obstacle'], rect)

    def display_robot(self, robot: Robot) -> None:
        """
        Renders the robot, including joints and links.
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.joint_1_pos())
        j2 = self.world.convert_to_display(robot.joint_2_pos())

        # Render joint 0
        pygame.draw.circle(self.screen, self.colors['robot'], j0, 4)
        # Render link 1
        pygame.draw.line(self.screen, self.colors['robot'], j0, j1, 2)
        # Render joint 1
        pygame.draw.circle(self.screen, self.colors['robot'], j1, 4)
        # Render link 2
        pygame.draw.line(self.screen, self.colors['robot'], j1, j2, 2)
        # Render joint 2
        pygame.draw.circle(self.screen, self.colors['robot'], j2, 4)

    def display_path(self, robot: Robot, path: List[Tuple[float, float]]) -> None:
        """
        Renders the planned path in the workspace.
        """
      
        points = [self.world.convert_to_display(robot.forward_kinematics(state.theta_0, state.theta_1)) for state in path]
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.colors['path'], False, points, 1)


    def update_display(self, robot: Robot, success: bool, goal: Tuple[float, float], path: Optional[List[Tuple[float, float]]] = None) -> bool:
        """
        Updates the display with the latest robot and world states.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(self.colors['background'])
        self.display_world(goal)
        if path:
            self.display_path(robot, path)
        self.display_robot(robot)

        if success:
            text = self.font.render("Success!", True, self.colors['success_text'])
            self.screen.blit(text, (10, 10))

        pygame.display.flip()
        return True

    def cleanup(self) -> None:
        """
        Cleans up pygame resources.
        """
        pygame.quit()


class Planner:
    """
    Abstract base class for motion planners.
    """
    def __init__(self, robot: Robot, world: World):
        self.robot = robot
        self.world = world

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        raise NotImplementedError


class GridBasedPlanner(Planner):
    """
    Base class for grid-based planners, providing common functionality.
    """
    def __init__(self, robot: Robot, world: World):
        super().__init__(robot, world)
        self.constants = robot.constants
        self.resolutions = (self.constants.THETA_0_RESOLUTION, self.constants.THETA_1_RESOLUTION)
        self.collision_cache: Dict[Tuple[float, float], bool] = {}

    def discretize_state(self, state: Tuple[float, float]) -> Tuple[float, float]:
        theta_0, theta_1 = state
        res_theta_0, res_theta_1 = self.resolutions
        discrete_theta_0 = round(theta_0 / res_theta_0) * res_theta_0
        discrete_theta_1 = round(theta_1 / res_theta_1) * res_theta_1
        return (discrete_theta_0, discrete_theta_1)

    def heuristic(self, current: Tuple[float, float], goal: Tuple[float, float]) -> float:
        # Euclidean distance in joint space
        return np.hypot(current[0] - goal[0], current[1] - goal[1])

    def distance(self, from_state: State, to_state: State) -> float:
        # Cost can be a combination of position and velocity changes
        delta_theta = np.hypot(to_state.theta_0 - from_state.theta_0, to_state.theta_1 - from_state.theta_1)
        delta_omega = np.hypot(to_state.omega_0 - from_state.omega_0, to_state.omega_1 - from_state.omega_1)
        # Weigh position and velocity changes as needed
        return delta_theta + delta_omega

    def within_joint_limits(self, node: Tuple[float, float]) -> bool:
        theta_0, theta_1 = node
        return (self.constants.JOINT_LIMITS[0] <= theta_0 <= self.constants.JOINT_LIMITS[1] and
                self.constants.JOINT_LIMITS[0] <= theta_1 <= self.constants.JOINT_LIMITS[1])

    def is_collision(self, node: Tuple[float, float]) -> bool:
        if node in self.collision_cache:
            return self.collision_cache[node]
        theta_0, theta_1 = node
        # Check for obstacle collisions
        if self.check_obstacle_collision(theta_0, theta_1):
            self.collision_cache[node] = True
            return True
        # Check for self-collisions
        if self.robot.self_collision(theta_0, theta_1):
            self.collision_cache[node] = True
            return True
        self.collision_cache[node] = False
        return False

    def is_goal(self, current_state: State, goal_state: State) -> bool:
        position_close = np.isclose(current_state.theta_0, goal_state.theta_0, atol=self.resolutions[0]) and \
                         np.isclose(current_state.theta_1, goal_state.theta_1, atol=self.resolutions[1])
        velocity_close = np.isclose(current_state.omega_0, goal_state.omega_0, atol=self.velocity_resolution) and \
                         np.isclose(current_state.omega_1, goal_state.omega_1, atol=self.velocity_resolution)
        return position_close and velocity_close


    def check_obstacle_collision(self, theta_0: float, theta_1: float) -> bool:
        link_segments = self.robot.compute_link_segments(theta_0, theta_1)
        for obstacle in self.world.obstacles:
            for segment in link_segments:
                if self.robot_collision(segment, obstacle):
                    return True
        return False

    def robot_collision(self, segment: Tuple[Tuple[float, float], Tuple[float, float]], obstacle: Obstacle) -> bool:
        p1, p2 = segment
        if obstacle.shape == 'circle':
            return CollisionChecker.line_circle_collision(p1, p2, obstacle.position, obstacle.size)
        elif obstacle.shape == 'rectangle':
            return CollisionChecker.line_rectangle_collision(p1, p2, obstacle.position, obstacle.size)
        return False

    def reconstruct_path(self, came_from: Dict[Tuple[float, float], Optional[Tuple[float, float]]],
                         current: Tuple[float, float]) -> List[Tuple[float, float]]:
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            if current is None:
                break
            total_path.append(current)
        return total_path[::-1]  # Reverse path


class AStarPlanner(GridBasedPlanner):
    """
    Optimized A* planner for the 2-link robotic arm.
    """
    def __init__(self, robot: Robot, world: World):
        super().__init__(robot, world)
        self.motion_primitives = self.generate_motion_primitives()

    def generate_motion_primitives(self) -> List[Tuple[float, float]]:
        max_alpha = self.constants.MAX_ACCELERATION
        min_alpha = -max_alpha
        num_accels = 3  # Adjust for desired granularity
        accels = np.linspace(min_alpha, max_alpha, num_accels)
        primitives = []
        for alpha_0 in accels:
            for alpha_1 in accels:
                primitives.append((alpha_0, alpha_1))
        return primitives

    def plan(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[State]:
        start_theta = self.robot.inverse_kinematics(*start_pos)
        goal_theta = self.robot.inverse_kinematics(*goal_pos)
        counter = itertools.count()

        # Initial velocities are zero
        start_state = State(*self.discretize_state(start_theta), 0.0, 0.0)
        goal_state = State(*self.discretize_state(goal_theta), 0.0, 0.0)  # Assuming zero velocity at goal

        open_set = []
        came_from: Dict[State, Optional[State]] = {}
        g_score = {start_state: 0}
        f_score = {start_state: self.heuristic(start_state, goal_state)}
        heapq.heappush(open_set, (f_score[start_state], next(counter), start_state))

        closed_set: Set[State] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if self.is_goal(current, goal_state):
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    if not self.is_collision(neighbor):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_state)
                        heapq.heappush(open_set, (f_score[neighbor], next(counter), neighbor))
                    else:
                        self.collision_cache[neighbor] = True

        raise ValueError("No path found")


    def get_neighbors(self, current_state: State) -> List[State]:
        neighbors = []
        for alpha_0, alpha_1 in self.motion_primitives:
            # Compute new velocities with acceleration constraints
            omega_0_new = current_state.omega_0 + alpha_0 * self.constants.DT
            omega_1_new = current_state.omega_1 + alpha_1 * self.constants.DT

            # Enforce velocity limits
            omega_0_new = np.clip(omega_0_new, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)
            omega_1_new = np.clip(omega_1_new, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)

            # Compute new positions
            theta_0_new = current_state.theta_0 + omega_0_new * self.constants.DT
            theta_1_new = current_state.theta_1 + omega_1_new * self.constants.DT

            # Discretize the state
            theta_0_new, theta_1_new = self.discretize_state((theta_0_new, theta_1_new))
            omega_0_new = round(omega_0_new / self.velocity_resolution) * self.VELOCITY_RESOLUTION
            omega_1_new = round(omega_1_new / self.velocity_resolution) * self.VELOCITY_RESOLUTION

            neighbor = State(theta_0_new, theta_1_new, omega_0_new, omega_1_new)
            if self.within_joint_limits((theta_0_new, theta_1_new)) and not self.is_collision(neighbor):
                neighbors.append(neighbor)
        return neighbors


class SBPLLatticePlanner(GridBasedPlanner):
    """
    SBPL-style lattice planner using optimized motion primitives with acceleration constraints.
    """
    def __init__(self, robot: Robot, world: World):
        super().__init__(robot, world)
        self.velocity_resolution = self.constants.VELOCITY_RESOLUTION
        self.motion_primitives = self.generate_motion_primitives()
        self.collision_cache: Dict[State, bool] = {}

    def generate_motion_primitives(self) -> List[Tuple[float, float]]:
        """
        Generate motion primitives based on allowable accelerations.
        """
        max_alpha = self.constants.MAX_ACCELERATION
        min_alpha = -max_alpha
        num_accels = 5  # Adjust based on desired granularity
        accelerations = np.linspace(min_alpha, max_alpha, num_accels)
        primitives = []
        for alpha_0 in accelerations:
            for alpha_1 in accelerations:
                primitives.append((alpha_0, alpha_1))
        return primitives

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[State]:
        start_theta = self.robot.inverse_kinematics(*start)
        goal_theta = self.robot.inverse_kinematics(*goal)
        counter = itertools.count()

        start_theta = self.discretize_state(start_theta)
        goal_theta = self.discretize_state(goal_theta)

        start_state = State(start_theta[0], start_theta[1], 0.0, 0.0)  # Initial velocities are zero
        goal_state = State(goal_theta[0], goal_theta[1], 0.0, 0.0)  # Assuming zero velocity at goal

        open_set = []
        came_from: Dict[State, Optional[State]] = {}
        g_score = {start_state: 0}
        f_score = {start_state: self.heuristic(start_state, goal_state)}
        heapq.heappush(open_set, (f_score[start_state], next(counter), start_state))

        closed_set: Set[State] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if self.is_goal(current, goal_state):
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    if not self.is_collision(neighbor):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_state)
                        heapq.heappush(open_set, (f_score[neighbor], next(counter), neighbor))
                    else:
                        self.collision_cache[neighbor] = True

        raise ValueError("No path found")

    def is_goal(self, current_state: State, goal_state: State) -> bool:
        position_close = np.isclose(current_state.theta_0, goal_state.theta_0, atol=self.resolutions[0]) and \
                         np.isclose(current_state.theta_1, goal_state.theta_1, atol=self.resolutions[1])
        velocity_close = np.isclose(current_state.omega_0, goal_state.omega_0, atol=self.velocity_resolution) and \
                         np.isclose(current_state.omega_1, goal_state.omega_1, atol=self.velocity_resolution)
        return position_close and velocity_close

    def get_neighbors(self, current_state: State) -> List[State]:
        neighbors = []
        for alpha_0, alpha_1 in self.motion_primitives:
            # Compute new velocities with acceleration constraints
            omega_0_new = current_state.omega_0 + alpha_0 * self.constants.DT
            omega_1_new = current_state.omega_1 + alpha_1 * self.constants.DT

            # Enforce velocity limits
            omega_0_new = np.clip(omega_0_new, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)
            omega_1_new = np.clip(omega_1_new, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)

            # Compute new positions
            theta_0_new = current_state.theta_0 + omega_0_new * self.constants.DT
            theta_1_new = current_state.theta_1 + omega_1_new * self.constants.DT

            # Discretize the state
            theta_0_new, theta_1_new = self.discretize_state((theta_0_new, theta_1_new))
            omega_0_new = round(omega_0_new / self.velocity_resolution) * self.velocity_resolution
            omega_1_new = round(omega_1_new / self.velocity_resolution) * self.velocity_resolution

            neighbor = State(theta_0_new, theta_1_new, omega_0_new, omega_1_new)
            if self.within_joint_limits((theta_0_new, theta_1_new)):
                neighbors.append(neighbor)
        return neighbors

    def is_collision(self, state: State) -> bool:
        if state in self.collision_cache:
            return self.collision_cache[state]
        theta_0, theta_1 = state.theta_0, state.theta_1
        # Check for obstacle collisions
        if self.check_obstacle_collision(theta_0, theta_1):
            self.collision_cache[state] = True
            return True
        # Check for self-collisions
        if self.robot.self_collision(theta_0, theta_1):
            self.collision_cache[state] = True
            return True
        self.collision_cache[state] = False
        return False

    def heuristic(self, current: State, goal: State) -> float:
        # Use Euclidean distance in joint space as heuristic
        return np.hypot(current.theta_0 - goal.theta_0, current.theta_1 - goal.theta_1)

    def distance(self, from_state: State, to_state: State) -> float:
        # Cost includes changes in positions and velocities
        delta_theta = np.hypot(to_state.theta_0 - from_state.theta_0, to_state.theta_1 - from_state.theta_1)
        delta_omega = np.hypot(to_state.omega_0 - from_state.omega_0, to_state.omega_1 - from_state.omega_1)
        return delta_theta + delta_omega  # You can adjust weights if needed

    def reconstruct_path(self, came_from: Dict[State, Optional[State]], current: State) -> List[State]:
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            if current is None:
                break
            total_path.append(current)
        return total_path[::-1]  # Reverse path



class Controller:
    """
    Follows a planned path provided by the planner.
    """
    def __init__(self, constants: RobotConstants, world: World, planner: Planner) -> None:
        self.constants = constants
        self.world = world
        self.planner = planner
        self.path: List[Tuple[float, float]] = []
        self.path_index = 0

    def set_goal(self, robot: Robot, goal: Tuple[float, float]) -> None:
        # Plan a path from the current position to the goal
        start_pos = robot.joint_2_pos()
        self.path = self.planner.plan(start_pos, goal)
        self.path_index = 0

    def step(self, robot: Robot) -> Robot:
        if self.path_index >= len(self.path):
            return robot  # Reached the end of the path

        next_state = self.path[self.path_index]

        # Compute desired accelerations
        desired_alpha_0 = (next_state.omega_0 - robot.omega_0) / self.constants.DT
        desired_alpha_1 = (next_state.omega_1 - robot.omega_1) / self.constants.DT

        # Clip accelerations to max acceleration
        max_alpha = self.constants.MAX_ACCELERATION
        alpha_0 = np.clip(desired_alpha_0, -max_alpha, max_alpha)
        alpha_1 = np.clip(desired_alpha_1, -max_alpha, max_alpha)

        # Update velocities
        robot.omega_0 += alpha_0 * self.constants.DT
        robot.omega_1 += alpha_1 * self.constants.DT

        # Enforce velocity limits
        robot.omega_0 = np.clip(robot.omega_0, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)
        robot.omega_1 = np.clip(robot.omega_1, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)

        # Update positions
        robot.theta_0 += robot.omega_0 * self.constants.DT
        robot.theta_1 += robot.omega_1 * self.constants.DT

        # Check if we have reached the next state in the path
        if np.isclose(robot.theta_0, next_state.theta_0, atol=self.constants.THETA_0_RESOLUTION) and \
           np.isclose(robot.theta_1, next_state.theta_1, atol=self.constants.THETA_1_RESOLUTION) and \
           np.isclose(robot.omega_0, next_state.omega_0, atol=self.constants.VELOCITY_RESOLUTION) and \
           np.isclose(robot.omega_1, next_state.omega_1, atol=self.constants.VELOCITY_RESOLUTION):
            self.path_index += 1  # Move to the next point in the path
        
        return robot



class Runner:
    """
    Manages the simulation loop, coordinating updates and visualization.
    """

    def __init__(self, robot: Robot, controller: Controller, world: World, visualizer: Visualizer) -> None:
        self.robot = robot
        self.controller = controller
        self.world = world
        self.visualizer = visualizer
        self.constants = robot.constants
        self.goal = generate_random_goal(
            self.constants.min_reachable_radius(),
            self.constants.max_reachable_radius()
        )

    def run(self) -> None:
        """
        Main simulation loop. Steps the controller and updates visualization.
        """
        running = True
        self.controller.set_goal(self.robot, self.goal)
        while running:
            self.robot = self.controller.step(self.robot)
            success = self.check_success(self.robot, self.goal)
            running = self.visualizer.update_display(self.robot, success, self.goal, path=self.controller.path)
            if success:
                # Generate a new random goal
                self.goal = generate_random_goal(
                    self.constants.min_reachable_radius(),
                    self.constants.max_reachable_radius()
                )
                self.controller.set_goal(self.robot, self.goal)
            time.sleep(self.constants.DT)

        # Add a pause before closing the visualizer
        pause = True
        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pause = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pause = False
            pygame.display.flip()
            pygame.time.wait(100)  # Wait for 100 milliseconds

    @staticmethod
    def check_success(robot: Robot, goal: Tuple[float, float]) -> bool:
        """
        Checks if the robot's end-effector is sufficiently close to the goal.
        """
        return np.allclose(robot.joint_2_pos(), goal, atol=0.5)

    def cleanup(self) -> None:
        """
        Cleans up resources used by the runner.
        """
        self.visualizer.cleanup()


def main() -> None:
    """
    Main entry point for the simulation.
    """
    # Load configuration
    with open('Config/config.json', 'r')  as f:
        config = json.load(f)

    # Initialize robot constants
    robot_constants = RobotConstants(config['robot'])

    # Initialize obstacles
    obstacles = [Obstacle(**obs) for obs in config['world'].get('obstacles', [])]

    # Generate a random rectangle obstacle and add it to the obstacles list
    workspace_size = min(config['world']['width'], config['world']['height'])
    min_distance = robot_constants.LINK_1  # For example, minimum distance is the length of the first link
    random_rectangle_obstacle = generate_random_rectangle_obstacle(workspace_size, min_distance)
    obstacles.append(random_rectangle_obstacle)

    # Initialize world
    world = World(
        config['world']['width'],
        config['world']['height'],
        tuple(config['world']['robot_origin']),
        obstacles
    )

    # Initialize robot
    robot = Robot(robot_constants)

    # Initialize planner
    # planner = AStarPlanner(robot, world)  # or
    planner = SBPLLatticePlanner(robot, world)

    # Initialize controller
    controller = Controller(robot_constants, world, planner)

    # Initialize visualizer
    visualizer = Visualizer(world, config)

    runner = Runner(robot, controller, world, visualizer)

    try:
        runner.run()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Simulation aborted: {e}")
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
