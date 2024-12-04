import time
from typing import List, Tuple, Union, Dict, Set, Optional
import numpy as np
import pygame
import json
import heapq
import time
from collections import deque
import itertools
import scipy.integrate

from collision_checker import CollisionChecker
from utils import generate_random_goal, generate_random_rectangle_obstacle
from obstacle import Obstacle
from dataclasses import dataclass
from enum import Enum

class TrajectoryType(Enum):
    ACCELERATION = 'acceleration'
    DECELERATION = 'deceleration'


@dataclass
class PathPoint:
    s: float  # Path parameter
    theta_0: float  # Joint angle 1
    theta_1: float  # Joint angle 2
    dtheta_0: float  # First derivative of theta_0 with respect to s
    dtheta_1: float  # First derivative of theta_1 with respect to s
    ddtheta_0: float  # Second derivative of theta_0 with respect to s
    ddtheta_1: float  # Second derivative of theta_1 with respect to s

@dataclass(frozen=True)
class State:
    theta_0: float
    theta_1: float
    omega_0: float
    omega_1: float

    def __lt__(self, other):
        # Required for heapq operations
        return (self.theta_0, self.theta_1, self.omega_0, self.omega_1) < (
            other.theta_0, other.theta_1, other.omega_0, other.omega_1)

class PlanningTimeoutError(Exception):
    """Raised when planning takes too long"""
    pass

class RobotConstants:
    def __init__(self, config):
        self.BASE_RADIUS = config.get('base_radius', 10.0)
        self.JOINT_LIMITS = tuple(config['joint_limits'])
        self.MAX_VELOCITY = config['max_velocity']
        self.MIN_VELOCITY = config['min_velocity']
        self.MAX_ACCELERATION = config['max_acceleration']
        self.MAX_JERK = config['max_jerk']
        self.VELOCITY_RESOLUTION = config['velocity_resolution']
        self.DT = config['dt']
        self.LINK_1 = config['link_lengths'][0]
        self.LINK_2 = config['link_lengths'][1]
        self.ROBOT_ORIGIN = tuple(config.get('robot_origin', (0.0, 0.0)))

        # Discretization parameters for planners
        self.THETA_0_RESOLUTION = config.get('theta_0_resolution', 0.1)
        self.THETA_1_RESOLUTION = config.get('theta_1_resolution', 0.1)
        self.CONSIDER_GRAVITY = config.get('consider_gravity', True)

    def min_reachable_radius(self) -> float:
        return max(self.LINK_1 - self.LINK_2, 0)

    def max_reachable_radius(self) -> float:
        return self.LINK_1 + self.LINK_2


class Robot:
    """
    Represents a two-link robotic arm with optional gravity consideration.
    """

    def __init__(self, constants: RobotConstants) -> None:
        self.constants = constants
        # Initialize angles and their histories
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.all_omega_0: List[float] = []
        self.all_omega_1: List[float] = []
        self.all_alpha_0: List[float] = []
        self.all_alpha_1: List[float] = []
        self._theta_0 = 0.0
        self._theta_1 = 0.0
        self.omega_0 = 0.0
        self.omega_1 = 0.0
        self.alpha_0 = 0.0
        self.alpha_1 = 0.0
        self.m1 = 2  # Mass of link 1
        self.m2 = 1.5  # Mass of link 2
        self.I1 = self.m1 * constants.LINK_1**2 / 12  # Inertia of link 1
        self.I2 = self.m2 * constants.LINK_2**2 / 12  # Inertia of link 2

        self.debug_helper = DebugHelper(debug=True)

    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self._theta_0 = value
        self.all_theta_0.append(value)
        self._validate_joint_limits(value, 0)

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self._theta_1 = value
        self.all_theta_1.append(value)
        self._validate_joint_limits(value, 1)

    def _validate_joint_limits(self, theta: float, joint_id: int) -> None:
        if not (self.constants.JOINT_LIMITS[0] <= theta <= self.constants.JOINT_LIMITS[1]):
            raise ValueError(f"Joint {joint_id} angle {theta} exceeds joint limits.")

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
        cos_theta_1 = (
            (x**2 + y**2 - self.constants.LINK_1**2 - self.constants.LINK_2**2)
            / (2 * self.constants.LINK_1 * self.constants.LINK_2)
        )
        cos_theta_1 = np.clip(cos_theta_1, -1.0, 1.0)
        theta_1_options = [np.arccos(cos_theta_1), -np.arccos(cos_theta_1)]
        solutions = []
        for theta_1 in theta_1_options:
            k1 = self.constants.LINK_1 + self.constants.LINK_2 * np.cos(theta_1)
            k2 = self.constants.LINK_2 * np.sin(theta_1)
            theta_0 = np.arctan2(y, x) - np.arctan2(k2, k1)
            solutions.append((theta_0, theta_1))

        # Choose the solution closest to current angles
        current_theta_0 = self.theta_0
        current_theta_1 = self.theta_1
        min_distance = float('inf')
        best_solution = solutions[0]
        for solution in solutions:
            distance = abs(solution[0] - current_theta_0) + abs(solution[1] - current_theta_1)
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
        return best_solution

    def compute_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dynamic matrices M, C, G with optional gravity consideration"""
        l1, l2 = self.constants.LINK_1, self.constants.LINK_2
        g = 9.8 if self.constants.CONSIDER_GRAVITY else 0.0  # Only apply gravity if enabled
        
        # Mass matrix (unchanged as it doesn't depend on gravity)
        M11 = (self.m1*l1**2/4 + self.I1 + self.m2*(l1**2 + l2**2/4 + l1*l2*np.cos(theta_1)))
        M12 = self.m2 * (l2**2/4 + l1*l2*np.cos(theta_1)/2)
        M21 = M12
        M22 = self.m2 * l2**2/4 + self.I2
        M = np.array([[M11, M12], [M21, M22]])
        
        # Coriolis matrix (unchanged as it depends on velocities, not gravity)
        h = self.m2 * l1 * l2 * np.sin(theta_1)
        C11 = -h * omega_1
        C12 = -h * (omega_0 + omega_1)
        C21 = h * omega_0
        C22 = 0
        C = np.array([[C11, C12], [C21, C22]])
        
        # Gravity vector (now conditionally applied)
        if self.constants.CONSIDER_GRAVITY:
            G1 = (self.m1*l1/2 + self.m2*l1)*g*np.cos(theta_0) + self.m2*l2*g*np.cos(theta_0 + theta_1)/2
            G2 = self.m2*l2*g*np.cos(theta_0 + theta_1)/2
        else:
            G1 = 0.0
            G2 = 0.0
        G = np.array([G1, G2])
        
        return M, C, G
        
    def forward_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float,
                        tau_0: float, tau_1: float) -> Tuple[float, float]:
        """Compute accelerations given torques"""
        M, C, G = self.compute_dynamics(theta_0, theta_1, omega_0, omega_1)
        omega = np.array([omega_0, omega_1])
        tau = np.array([tau_0, tau_1])
        
        # Solve M * alpha = tau - C * omega - G
        alpha = np.linalg.solve(M, tau - C @ omega - G)
        return alpha[0], alpha[1]
        
    def inverse_dynamics(self, theta_0: float, theta_1: float, omega_0: float, omega_1: float,
                        alpha_0: float, alpha_1: float) -> Tuple[float, float]:
        """Compute required torques for desired accelerations"""
        M, C, G = self.compute_dynamics(theta_0, theta_1, omega_0, omega_1)
        omega = np.array([omega_0, omega_1])
        alpha = np.array([alpha_0, alpha_1])
        
        # tau = M * alpha + C * omega + G
        tau = M @ alpha + C @ omega + G
        return tau[0], tau[1]


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

        self.debug_helper = DebugHelper(debug=True)

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
        self.screen_width = world.width * 2  # Double the width for graphs
        self.screen_height = world.height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Simulation")
        self.font = pygame.font.SysFont(None, 24)
        self.colors = {
            'background': tuple(config['visualizer']['colors']['background']),
            'robot': tuple(config['visualizer']['colors']['robot']),
            'goal': tuple(config['visualizer']['colors']['goal']),
            'obstacle': tuple(config['visualizer']['colors']['obstacle']),
            'success_text': tuple(config['visualizer']['colors']['success_text']),
            'path': tuple(config['visualizer']['colors'].get('path', [0, 0, 255])),
            'reference_path': (0, 255, 0),
            'actual_path': (255, 0, 0),
            'reference': (0, 255, 0),
            'actual': (255, 0, 0)
        }

        self.debug_helper = DebugHelper(debug=True)

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

    def display_paths(self, robot: Robot, controller: 'Controller') -> None:
        """
        Renders both the planned path (reference) and the actual path followed by the robot.
        """
        # Reference path (planned path)
        if controller.path:
            reference_points = [self.world.convert_to_display(robot.forward_kinematics(state.theta_0, state.theta_1)) for state in controller.path]
            if len(reference_points) > 1:
                pygame.draw.lines(self.screen, self.colors['reference_path'], False, reference_points, 2)

        # Actual path
        if controller.actual_theta_0 and controller.actual_theta_1:
            actual_points = [self.world.convert_to_display(robot.forward_kinematics(theta_0, theta_1)) for theta_0, theta_1 in zip(controller.actual_theta_0, controller.actual_theta_1)]
            if len(actual_points) > 1:
                pygame.draw.lines(self.screen, self.colors['actual_path'], False, actual_points, 2)

    def update_display(self, robot: Robot, success: bool, goal: Tuple[float, float], controller: Optional['Controller'] = None) -> bool:
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
        if controller is not None:
            self.display_paths(robot, controller)
        self.display_robot(robot)

        if success:
            success_text = self.font.render("Goal Reached Successfully!", True, self.colors['success_text'])
            text_rect = success_text.get_rect(center=(self.world.width // 4, 30))
            self.screen.blit(success_text, text_rect)

        if controller is not None:
            self.display_graphs(controller)

        pygame.display.flip()
        return True

    def display_graphs(self, controller: 'Controller') -> None:
        """Displays velocity and acceleration graphs with improved error handling"""
        try:
            graph_area_width = self.screen_width // 2
            graph_area_height = self.screen_height

            # Define rectangles for graphs
            graph_height = graph_area_height // 4
            velocity_0_rect = pygame.Rect(self.screen_width // 2, 0, graph_area_width, graph_height)
            velocity_1_rect = pygame.Rect(self.screen_width // 2, graph_height, graph_area_width, graph_height)
            acceleration_0_rect = pygame.Rect(self.screen_width // 2, 2 * graph_height, graph_area_width, graph_height)
            acceleration_1_rect = pygame.Rect(self.screen_width // 2, 3 * graph_height, graph_area_width, graph_height)

            # Draw graph backgrounds
            pygame.draw.rect(self.screen, (50, 50, 50), velocity_0_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), velocity_1_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), acceleration_0_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), acceleration_1_rect)

            # Plot velocities for joint 0
            if hasattr(controller, 'reference_omega_0') and hasattr(controller, 'actual_omega_0'):
                reference_velocity_0 = list(controller.reference_omega_0)  # Convert to list if needed
                actual_velocity_0 = list(controller.actual_omega_0)

                # Ensure we have valid data
                if reference_velocity_0 or actual_velocity_0:
                    # Compute min/max while handling empty lists
                    all_vals = [v for v in reference_velocity_0 + actual_velocity_0 if isinstance(v, (int, float))]
                    if all_vals:
                        min_val = min(all_vals)
                        max_val = max(all_vals)
                        # Add padding to prevent scaling issues
                        padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                        min_velocity_0 = min_val - padding
                        max_velocity_0 = max_val + padding
                        
                        self.plot_graph(reference_velocity_0, self.colors['reference'], 
                                      velocity_0_rect, min_velocity_0, max_velocity_0, "Velocity Joint 0")
                        self.plot_graph(actual_velocity_0, self.colors['actual'], 
                                      velocity_0_rect, min_velocity_0, max_velocity_0)

            # Plot velocities for joint 1
            if hasattr(controller, 'reference_omega_1') and hasattr(controller, 'actual_omega_1'):
                reference_velocity_1 = list(controller.reference_omega_1)
                actual_velocity_1 = list(controller.actual_omega_1)

                # Ensure we have valid data
                if reference_velocity_1 or actual_velocity_1:
                    # Compute min/max while handling empty lists
                    all_vals = [v for v in reference_velocity_1 + actual_velocity_1 if isinstance(v, (int, float))]
                    if all_vals:
                        min_val = min(all_vals)
                        max_val = max(all_vals)
                        # Add padding to prevent scaling issues
                        padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                        min_velocity_1 = min_val - padding
                        max_velocity_1 = max_val + padding
                        
                        self.plot_graph(reference_velocity_1, self.colors['reference'], 
                                      velocity_1_rect, min_velocity_1, max_velocity_1, "Velocity Joint 1")
                        self.plot_graph(actual_velocity_1, self.colors['actual'], 
                                      velocity_1_rect, min_velocity_1, max_velocity_1)

            # Plot accelerations with error checking
            if (hasattr(controller, 'reference_alpha_0') and hasattr(controller, 'actual_alpha_0') and
                controller.reference_alpha_0 and controller.actual_alpha_0):
                
                # Joint 0 acceleration
                all_acceleration_0 = controller.reference_alpha_0 + controller.actual_alpha_0
                min_acceleration_0 = min(all_acceleration_0) if all_acceleration_0 else -1
                max_acceleration_0 = max(all_acceleration_0) if all_acceleration_0 else 1
                
                # Add padding to prevent divide by zero
                if abs(max_acceleration_0 - min_acceleration_0) < 0.1:
                    min_acceleration_0 -= 0.1
                    max_acceleration_0 += 0.1
                
                self.plot_graph(controller.reference_alpha_0, self.colors['reference'], 
                              acceleration_0_rect, min_acceleration_0, max_acceleration_0, 
                              "Acceleration Joint 0")
                self.plot_graph(controller.actual_alpha_0, self.colors['actual'], 
                              acceleration_0_rect, min_acceleration_0, max_acceleration_0)

                # Joint 1 acceleration
                all_acceleration_1 = controller.reference_alpha_1 + controller.actual_alpha_1
                min_acceleration_1 = min(all_acceleration_1) if all_acceleration_1 else -1
                max_acceleration_1 = max(all_acceleration_1) if all_acceleration_1 else 1
                
                # Add padding to prevent divide by zero
                if abs(max_acceleration_1 - min_acceleration_1) < 0.1:
                    min_acceleration_1 -= 0.1
                    max_acceleration_1 += 0.1
                
                self.plot_graph(controller.reference_alpha_1, self.colors['reference'], 
                              acceleration_1_rect, min_acceleration_1, max_acceleration_1, 
                              "Acceleration Joint 1")
                self.plot_graph(controller.actual_alpha_1, self.colors['actual'], 
                              acceleration_1_rect, min_acceleration_1, max_acceleration_1)

            # Draw Legends
            self.draw_legends()
            
        except Exception as e:
            print(f"Error in display_graphs: {e}")

    def plot_graph(self, data, color, rect, min_value, max_value, label=None):
        """Plot a graph with improved error handling and scaling"""
        try:
            if not data or len(data) < 2:
                return

            # Ensure valid scaling
            value_range = max_value - min_value
            if value_range == 0:
                max_value += 0.1
                min_value -= 0.1
                value_range = 0.2

            x_scale = rect.width / (len(data) - 1)
            y_scale = rect.height / value_range

            # Filter and validate points
            points = []
            for i, value in enumerate(data):
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    x = rect.left + i * x_scale
                    y = rect.bottom - ((value - min_value) * y_scale)
                    # Clamp y value to rect bounds
                    y = max(rect.top, min(rect.bottom, y))
                    points.append((x, y))

            # Draw lines if we have valid points
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 1)

            # Draw label if provided
            if label:
                label_surf = self.font.render(label, True, (255, 255, 255))
                self.screen.blit(label_surf, (rect.left + 5, rect.top + 5))

        except Exception as e:
            print(f"Error in plot_graph: {e}")

    def draw_legends(self):
        """
        Draw legends for the graphs.
        """
        legend_x = self.screen_width - 150
        legend_y = 10
        legend_spacing = 20

        # Reference Legend
        pygame.draw.line(self.screen, self.colors['reference'], (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        ref_text = self.font.render("Reference", True, self.colors['reference'])
        self.screen.blit(ref_text, (legend_x + 30, legend_y - 8))

        # Actual Legend
        legend_y += legend_spacing
        pygame.draw.line(self.screen, self.colors['actual'], (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        act_text = self.font.render("Actual", True, self.colors['actual'])
        self.screen.blit(act_text, (legend_x + 30, legend_y - 8))

    def cleanup(self) -> None:
        """
        Cleans up pygame resources.
        """
        pygame.quit()

class Planner:
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot: Robot, final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """
        Plans a path from start to goal with desired final velocities.
        """
        raise NotImplementedError

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot, 
            final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List['State']:
        """
        Plans a path from start to goal with desired final velocities.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            robot: Robot instance
            final_velocities: Desired final velocities (omega_0, omega_1)
            
        Returns:
            List of States representing the planned path
            
        Raises:
            ValueError: If no path is found
            PlanningTimeoutError: If planning takes too long
        """
        raise NotImplementedError


class GridBasedPlanner(Planner):
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
        self.constants = robot.constants
        self.resolutions = (self.constants.THETA_0_RESOLUTION, self.constants.THETA_1_RESOLUTION)
        self.collision_cache: Dict[State, bool] = {}
        self.max_cache_size = 10000
        self.planning_timeout = 30.0
        self.progress_timeout = 15.0
        self.min_progress_threshold = 0.1  
        self.max_explored_states = 10000

    def clear_cache(self):
        """Clear the collision cache if it exceeds the maximum size"""
        if len(self.collision_cache) > self.max_cache_size:
            # Keep only the most recent entries
            items = list(self.collision_cache.items())
            self.collision_cache = dict(items[-self.max_cache_size:])

    def discretize_state(self, state: Tuple[float, float]) -> Tuple[float, float]:
        theta_0, theta_1 = state
        
        # Check distance to nearest obstacle
        current_pos = self.robot.forward_kinematics(theta_0, theta_1)
        min_obstacle_dist = self._get_min_obstacle_distance(current_pos)
        
        # Adjust resolution based on obstacle proximity
        base_resolution = self.resolutions[0]
        if min_obstacle_dist < 10.0:
            resolution = base_resolution / 2  # Finer resolution near obstacles
        else:
            resolution = base_resolution * 2  # Coarser resolution in free space
            
        discrete_theta_0 = round(theta_0 / resolution) * resolution
        discrete_theta_1 = round(theta_1 / resolution) * resolution
        
        return (discrete_theta_0, discrete_theta_1)

    def _get_min_obstacle_distance(self, pos: Tuple[float, float]) -> float:
        min_dist = float('inf')
        for obstacle in self.world.obstacles:
            if obstacle.shape == 'circle':
                dist = np.hypot(pos[0] - obstacle.position[0], 
                              pos[1] - obstacle.position[1]) - obstacle.size
            else:
                # Rectangle distance calculation
                dx = max(abs(pos[0] - obstacle.position[0]) - obstacle.size[0]/2, 0)
                dy = max(abs(pos[1] - obstacle.position[1]) - obstacle.size[1]/2, 0)
                dist = np.hypot(dx, dy)
            min_dist = min(min_dist, dist)
        return min_dist

    def heuristic(self, current: State, goal: State) -> float:
        """Modified heuristic with workspace and configuration space components"""
        # Get end-effector positions
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        
        # Workspace distance
        workspace_distance = np.hypot(current_pos[0] - goal_pos[0], 
                                    current_pos[1] - goal_pos[1])
        
        # Configuration space distance
        config_distance = np.hypot(current.theta_0 - goal.theta_0, 
                                 current.theta_1 - goal.theta_1)
        
        # Velocity alignment penalty
        velocity_alignment = np.hypot(current.omega_0 - goal.omega_0,
                                    current.omega_1 - goal.omega_1)
        
        # Combined heuristic with tuned weights
        w1, w2, w3 = 1.0, 0.3, 0.1  # Emphasize workspace distance
        return w1 * workspace_distance + w2 * config_distance + w3 * velocity_alignment

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

    def is_collision(self, state: State) -> bool:
        if state in self.collision_cache:
            return self.collision_cache[state]
        theta_0 = state.theta_0
        theta_1 = state.theta_1
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

    def is_goal(self, current_state: State, goal_state: State) -> bool:
        # Check end-effector position instead of joint angles
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(goal_state.theta_0, goal_state.theta_1)
        
        position_threshold = 0.5  # End-effector position threshold
        velocity_threshold = self.constants.VELOCITY_RESOLUTION
        
        position_close = np.hypot(current_pos[0] - goal_pos[0], 
                                current_pos[1] - goal_pos[1]) <= position_threshold
        
        velocity_close = (abs(current_state.omega_0 - goal_state.omega_0) <= velocity_threshold and
                         abs(current_state.omega_1 - goal_state.omega_1) <= velocity_threshold)
        
        return position_close #and velocity_close  # Require both position and velocity to be close


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

    def reconstruct_path(self, came_from: Dict[State, Optional[State]], current: State) -> List[State]:
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
    def __init__(self, robot, world, mode='kinodynamic'):
        super().__init__(robot, world)
        self.mode = mode  # 'kinodynamic' or 'coarse'
        self.velocity_resolution = self.constants.VELOCITY_RESOLUTION
        self.max_smoothing_error = 0.1  # Maximum allowed deviation during smoothing
        self.min_path_points = 5  # Minimum points to maintain path fidelity
        self.motion_primitives = self._generate_motion_primitives() if mode == 'kinodynamic' else None
        self.last_progress_time = 0

        # Smoothness parameters
        self.max_jerk = self.constants.MAX_JERK  # Maximum allowed jerk
        self.smoothing_window = 5  # Window size for smoothing
        self.min_velocity_threshold = 0.1  # Minimum velocity for smoothing
        self.acceleration_weight = 0.3  # Weight for acceleration cost
        self.jerk_weight = 0.2  # Weight for jerk cost
        
        # Add debug helper
        self.debug_helper = DebugHelper(debug=True)



    def set_mode(self, mode: str) -> None:
        """Switch between planning modes"""
        if mode not in ['kinodynamic', 'coarse']:
            raise ValueError("Mode must be either 'kinodynamic' or 'coarse'")
        self.mode = mode
        self.motion_primitives = self._generate_motion_primitives() if mode == 'kinodynamic' else None

    def get_neighbors_coarse(self, current_state: State) -> List[State]:
        """Get neighbors for coarse planning with improved adaptive step size and directional sampling"""
        neighbors = []
        
        # Get current end-effector position and goal position
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(self.goal_state.theta_0, self.goal_state.theta_1)
        
        # Compute distance to goal
        dist_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
        
        # More adaptive resolution based on distance to goal
        base_resolution = self.constants.THETA_0_RESOLUTION
        if dist_to_goal < 1.0:
            theta_resolution = base_resolution * 0.5  # Finer resolution near goal
        elif dist_to_goal < 5.0:
            theta_resolution = base_resolution * 2.0
        else:
            theta_resolution = base_resolution * 3.0
            
        # Generate more diverse angle steps
        # angle_steps = [
        #     -3.0 * theta_resolution,
        #     -1.5 * theta_resolution, -0.25 * theta_resolution, 0.25 * theta_resolution, 1.5 * theta_resolution,
        #      3.0 * theta_resolution
        # ]
        angle_steps = [
            -1.5 * theta_resolution, -0.25 * theta_resolution, 0.25 * theta_resolution, 1.5 * theta_resolution,
        ]
        # Add directional bias towards goal with improved Jacobian handling
        goal_direction = np.array([goal_pos[0] - current_pos[0], goal_pos[1] - current_pos[1]])
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
            
            jacobian = self._compute_jacobian(current_state.theta_0, current_state.theta_1)
            if jacobian is not None:
                try:
                    # Use damped least squares for more stable inverse
                    lambda_factor = 0.1
                    J_T = jacobian.T
                    inv_term = np.linalg.inv(jacobian @ J_T + lambda_factor * np.eye(2))
                    joint_direction = J_T @ inv_term @ goal_direction
                    
                    # Add more varied biased steps
                    bias_scales = [0.5,  1.5, 2.5, 3.5]
                    biased_steps0 = [joint_direction[0] * scale * theta_resolution 
                                   for scale in bias_scales]
                    biased_steps1 = [joint_direction[1] * scale * theta_resolution 
                                   for scale in bias_scales]
                    angle_steps.extend(biased_steps0)
                    angle_steps.extend(biased_steps1)
                except np.linalg.LinAlgError:
                    pass
        
        # Generate neighbors with improved sampling
        for d_theta0 in angle_steps:
            for d_theta1 in angle_steps:
                theta_0_new = current_state.theta_0 + d_theta0
                theta_1_new = current_state.theta_1 + d_theta1

                # Skip if too similar to current state
                if abs(d_theta0) < 1e-6 and abs(d_theta1) < 1e-6:
                    continue

                # Check joint limits
                if not self.within_joint_limits((theta_0_new, theta_1_new)):
                    continue

                # Create neighbor state
                neighbor = State(theta_0_new, theta_1_new, 0.0, 0.0)

                # Check collision and workspace constraints
                if not self.is_collision(neighbor):
                    neighbor_pos = self.robot.forward_kinematics(theta_0_new, theta_1_new)
                    if (self._is_in_workspace(neighbor_pos) and 
                        self._check_step_feasibility(current_pos, neighbor_pos)):
                        neighbors.append(neighbor)

        # Debug output
        if len(neighbors) == 0:
            self.debug_helper.log_state(f"Warning: No valid neighbors found at distance {dist_to_goal:.3f}")

        return neighbors

    def _compute_jacobian(self, theta_0: float, theta_1: float) -> Optional[np.ndarray]:
        """Compute the Jacobian matrix for the robot arm"""
        l1, l2 = self.robot.constants.LINK_1, self.robot.constants.LINK_2
        
        # Compute trigonometric terms
        c1 = np.cos(theta_0)
        s1 = np.sin(theta_0)
        c12 = np.cos(theta_0 + theta_1)
        s12 = np.sin(theta_0 + theta_1)
        
        # Compute Jacobian elements
        J = np.array([
            [-l1*s1 - l2*s12, -l2*s12],
            [l1*c1 + l2*c12, l2*c12]
        ])
        
        return J

    def _check_step_feasibility(self, current_pos: Tuple[float, float], 
                              next_pos: Tuple[float, float], 
                              max_step: float = 0.5) -> bool:
        """Check if the step between positions is feasible"""
        step_size = np.hypot(next_pos[0] - current_pos[0], 
                           next_pos[1] - current_pos[1])
        return step_size <= max_step
    def _is_in_workspace(self, pos: Tuple[float, float]) -> bool:
        """Check if a position is within the robot's workspace"""
        dist_from_base = np.hypot(pos[0], pos[1])
        min_reach = self.robot.constants.min_reachable_radius()
        max_reach = self.robot.constants.max_reachable_radius()
        return min_reach <= dist_from_base <= max_reach

    def heuristic_coarse(self, current: State, goal: State) -> float:
        """Simplified heuristic for coarse planning"""
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        return np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])

    def distance_coarse(self, from_state: State, to_state: State) -> float:
        """Simple Euclidean distance in joint space for coarse planning"""
        return np.hypot(to_state.theta_0 - from_state.theta_0,
                       to_state.theta_1 - from_state.theta_1)

    def is_goal_coarse(self, current_state: State, goal_state: State) -> bool:
        """Improved goal check for coarse planning with adaptive thresholds"""
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(goal_state.theta_0, goal_state.theta_1)
        distance_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
        
        # Adaptive position threshold based on robot dimensions
        l1, l2 = self.robot.constants.LINK_1, self.robot.constants.LINK_2
        workspace_size = l1 + l2
        position_threshold = min(0.2, workspace_size * 0.05)  # 5% of workspace size or 0.2, whichever is smaller
        
        # Adaptive angle threshold based on distance
        base_angle_threshold = 0.15  # radians (~8.6 degrees)
        angle_threshold = base_angle_threshold * (1.0 + distance_to_goal / workspace_size)
        
        angle_diff_0 = abs(current_state.theta_0 - goal_state.theta_0)
        angle_diff_1 = abs(current_state.theta_1 - goal_state.theta_1)
        
        # Debug output
        # self.debug_helper.log_state(f"Goal check - distance: {distance_to_goal:.3f}, threshold: {position_threshold:.3f}")
        # self.debug_helper.log_state(f"Angle diffs - theta0: {angle_diff_0:.3f}, theta1: {angle_diff_1:.3f}, threshold: {angle_threshold:.3f}")
        
        return (distance_to_goal <= position_threshold) #and 
               # angle_diff_0 <= angle_threshold and 
               # angle_diff_1 <= angle_threshold)

    def validate_sanitized_path(self, original_path: List[State], sanitized_path: List[State]) -> bool:
        """
        Validate that the sanitized path maintains the essential properties of the original path.
        
        Args:
            original_path (List[State]): Original path before sanitization
            sanitized_path (List[State]): Path after sanitization
            
        Returns:
            bool: True if sanitized path is valid, False otherwise
        """
        if not original_path or not sanitized_path:
            return False
            
        # Check start and goal states match
        if (sanitized_path[0] != original_path[0] or 
            sanitized_path[-1] != original_path[-1]):
            return False
        
        # Check maximum deviation between paths
        max_deviation = 0.0
        for orig_state in original_path:
            # Find closest point in sanitized path
            min_dist = float('inf')
            for san_state in sanitized_path:
                dist = np.hypot(
                    orig_state.theta_0 - san_state.theta_0,
                    orig_state.theta_1 - san_state.theta_1
                )
                min_dist = min(min_dist, dist)
            max_deviation = max(max_deviation, min_dist)
        
        # Maximum allowed deviation (in radians)
        max_allowed_deviation = 0.2
        if max_deviation > max_allowed_deviation:
            return False
        
        # Check for velocity discontinuities
        for i in range(1, len(sanitized_path)):
            delta_v = np.hypot(
                sanitized_path[i].omega_0 - sanitized_path[i-1].omega_0,
                sanitized_path[i].omega_1 - sanitized_path[i-1].omega_1
            )
            if delta_v > self.constants.MAX_VELOCITY:
                return False
        
        return True
    def check_timeout_and_progress(self, current: State, goal: State, start_time: float):
        """Check for timeout and lack of progress"""
        current_time = time.time()
        if current_time - start_time > self.planning_timeout:
            raise PlanningTimeoutError("Planning timeout reached")

        # Check progress
        current_distance = self.heuristic(current, goal)
        if current_distance < self.best_distance_to_goal:
            self.best_distance_to_goal = current_distance
            self.last_progress_time = current_time
        elif current_time - self.last_progress_time > self.progress_timeout:  # 25 seconds without progress
            raise PlanningTimeoutError("No progress made in planning")

    def _generate_motion_primitives(self) -> List[Tuple[float, float]]:
        """Generate motion primitives with better goal-reaching characteristics"""
        max_alpha = self.constants.MAX_ACCELERATION
        dt = self.constants.DT
        
        # Keep acceleration levels more conservative
        coarse_levels = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
        acceleration_levels = [max_alpha * factor for factor in coarse_levels]
        
        primitives = []
        for alpha_0 in acceleration_levels:
            for alpha_1 in acceleration_levels:
                primitives.append((alpha_0, alpha_1))
        
        return primitives

    def optimize_path(self, path: List[State]) -> List[State]:
        """Post-process the path to improve smoothness and physics consistency"""
        if len(path) < 3:
            return path

        # First apply basic smoothing
        smoothed_path = self._smooth_path(path)
        
        # Now optimize velocities for smooth transitions
        optimized_path = []
        dt = self.constants.DT
        
        # Keep start state
        optimized_path.append(smoothed_path[0])
        
        for i in range(1, len(smoothed_path) - 1):
            prev_state = optimized_path[-1]
            current_state = smoothed_path[i]
            next_state = smoothed_path[i + 1]
            
            # Compute optimal velocities based on position differences
            delta_theta_0 = next_state.theta_0 - prev_state.theta_0
            delta_theta_1 = next_state.theta_1 - prev_state.theta_1
            
            # Compute time-optimal velocities
            target_omega_0 = delta_theta_0 / (2 * dt)
            target_omega_1 = delta_theta_1 / (2 * dt)
            
            # Apply velocity limits
            max_vel = self.constants.MAX_VELOCITY
            omega_0 = np.clip(target_omega_0, -max_vel, max_vel)
            omega_1 = np.clip(target_omega_1, -max_vel, max_vel)
            
            # Create optimized state
            optimized_state = State(
                current_state.theta_0,
                current_state.theta_1,
                omega_0,
                omega_1
            )
            
            # Verify dynamic feasibility
            if self._check_dynamic_constraints(
                optimized_state.theta_0, optimized_state.theta_1,
                optimized_state.omega_0, optimized_state.omega_1
            ):
                optimized_path.append(optimized_state)
            else:
                optimized_path.append(current_state)
        
        # Keep goal state
        optimized_path.append(smoothed_path[-1])
        
        # Apply final velocity smoothing
        final_path = []
        window_size = 3
        for i in range(len(optimized_path)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(optimized_path), i + window_size//2)
            window = optimized_path[start_idx:end_idx]
            
            # Smooth velocities
            omega_0 = sum(s.omega_0 for s in window) / len(window)
            omega_1 = sum(s.omega_1 for s in window) / len(window)
            
            final_path.append(State(
                optimized_path[i].theta_0,
                optimized_path[i].theta_1,
                omega_0,
                omega_1
            ))
        return final_path

    def _optimize_velocities(self, path: List[State]) -> List[State]:
        """Optimize velocities while preserving path geometry"""
        if len(path) < 3:
            return path
            
        optimized_path = [path[0]]  # Keep start state
        dt = self.constants.DT
        
        for i in range(1, len(path) - 1):
            current = path[i]
            next_state = path[i + 1]
            prev_state = optimized_path[-1]
            
            # Compute velocity based on position difference
            delta_theta_0 = next_state.theta_0 - prev_state.theta_0
            delta_theta_1 = next_state.theta_1 - prev_state.theta_1
            
            # Keep velocities conservative
            omega_0 = np.clip(delta_theta_0 / (2 * dt), -self.constants.MAX_VELOCITY * 0.8, 
                            self.constants.MAX_VELOCITY * 0.8)
            omega_1 = np.clip(delta_theta_1 / (2 * dt), -self.constants.MAX_VELOCITY * 0.8,
                            self.constants.MAX_VELOCITY * 0.8)
            
            optimized_path.append(State(
                current.theta_0,
                current.theta_1,
                omega_0,
                omega_1
            ))
        
        optimized_path.append(path[-1])  # Keep goal state
        return optimized_path   

    def _check_dynamic_constraints(self, theta_0: float, theta_1: float, 
                                 omega_0: float, omega_1: float) -> bool:
        """Verify dynamic constraints including centripetal forces"""
        # Check velocity limits
        if abs(omega_0) > self.constants.MAX_VELOCITY or abs(omega_1) > self.constants.MAX_VELOCITY:
            return False
            
        # Check centripetal acceleration
        l1, l2 = self.robot.constants.LINK_1, self.robot.constants.LINK_2
        # Compute centripetal acceleration for each link
        a_cent_1 = omega_0**2 * l1  # First link
        a_cent_2 = (omega_0 + omega_1)**2 * l2  # Second link
        
        # Maximum allowable centripetal acceleration (tunable parameter)
        max_centripetal = self.constants.MAX_ACCELERATION * 1.5
        
        if a_cent_1 > max_centripetal or a_cent_2 > max_centripetal:
            return False
            
        return True

    def _smooth_path(self, path: List[State]) -> List[State]:
        """Apply path smoothing to reduce jerky motions"""
        if len(path) < 3:
            return path
            
        smoothed_path = [path[0]]  # Keep start state
        window_size = 5
        
        for i in range(1, len(path) - 1):
            # Get window of states for smoothing
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(path), i + window_size//2)
            window = path[start_idx:end_idx]
            
            # Compute smoothed state
            theta_0 = sum(s.theta_0 for s in window) / len(window)
            theta_1 = sum(s.theta_1 for s in window) / len(window)
            omega_0 = sum(s.omega_0 for s in window) / len(window)
            omega_1 = sum(s.omega_1 for s in window) / len(window)
            
            # Create smoothed state
            smoothed_state = State(theta_0, theta_1, omega_0, omega_1)
            
            # Verify smoothed state is valid
            if (self.within_joint_limits((theta_0, theta_1)) and 
                not self.is_collision(smoothed_state)):
                smoothed_path.append(smoothed_state)
            else:
                smoothed_path.append(path[i])
                
        smoothed_path.append(path[-1])  # Keep goal state
        return smoothed_path


    def _check_kinematic_feasibility(self, current: State, next: State) -> bool:
        """Verify kinematic feasibility with relaxed constraints and debugging"""
        dt = self.constants.DT
        max_accel = self.constants.MAX_ACCELERATION
        
        # Compute required accelerations
        alpha_0 = (next.omega_0 - current.omega_0) / dt
        alpha_1 = (next.omega_1 - current.omega_1) / dt
        
        # self.debug_helper.log_state(f"\nKinematic Feasibility Check:")
        # self.debug_helper.log_state(f"Current velocities: ω0={current.omega_0:.3f}, ω1={current.omega_1:.3f}")
        # self.debug_helper.log_state(f"Next velocities: ω0={next.omega_0:.3f}, ω1={next.omega_1:.3f}")
        # self.debug_helper.log_state(f"Required accelerations: α0={alpha_0:.3f}, α1={alpha_1:.3f}")
        # self.debug_helper.log_state(f"Max allowed acceleration: {max_accel}")
        
        # Allow slightly higher accelerations for better maneuverability
        accel_tolerance = 1.5  # Increase from 1.1 to 1.5
        max_allowed = max_accel * accel_tolerance
        
        if abs(alpha_0) > max_allowed or abs(alpha_1) > max_allowed:
            # self.debug_helper.log_state(f"Failed: Required accelerations exceed limits")
            return False
            
        # Add velocity change limit
        max_vel_change = self.constants.MAX_VELOCITY * 0.5  # Limit velocity changes to 50% of max
        if abs(next.omega_0 - current.omega_0) > max_vel_change or \
           abs(next.omega_1 - current.omega_1) > max_vel_change:
            # self.debug_helper.log_state(f"Failed: Velocity change too large")
            return False
        
        # self.debug_helper.log_state("Passed kinematic feasibility")
        return True

    def _generate_random_primitives(self) -> List[Tuple[float, float]]:
        max_alpha = self.constants.MAX_ACCELERATION
        num_random = 10  # Number of random primitives to generate
        primitives = []
        
        for _ in range(num_random):
            alpha_0 = np.random.uniform(-max_alpha, max_alpha)
            alpha_1 = np.random.uniform(-max_alpha, max_alpha)
            primitives.append((alpha_0, alpha_1))
        
        return primitives

    def get_neighbors(self, current_state: State) -> List[State]:
        """Generate neighbors with better goal convergence"""
        neighbors = []
        dt = self.constants.DT
        max_vel = self.constants.MAX_VELOCITY
        min_vel = self.constants.MIN_VELOCITY
        resolution = self.constants.THETA_0_RESOLUTION

        # Calculate distance to goal
        current_pos = self.robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
        goal_pos = self.robot.forward_kinematics(self.goal_state.theta_0, self.goal_state.theta_1)
        dist_to_goal = np.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])

         # Deceleration profile
        decel_start_dist = 2.0
        if dist_to_goal < decel_start_dist:
            vel_scale = min(1.0, dist_to_goal / decel_start_dist)
            max_vel *= vel_scale
            min_vel *= vel_scale
        
        for alpha_0, alpha_1 in self.motion_primitives:

            # Scale acceleration based on distance to goal
            if dist_to_goal < decel_start_dist:
                alpha_0 *= vel_scale
                alpha_1 *= vel_scale

            # Forward simulate with acceleration limits
            omega_0_new = np.clip(
                current_state.omega_0 + alpha_0 * dt,
                -max_vel, max_vel
            )
            omega_1_new = np.clip(
                current_state.omega_1 + alpha_1 * dt,
                -max_vel, max_vel
            )

            # Enforce minimum velocity if joint is moving
            if abs(current_state.theta_0 - self.goal_state.theta_0) > 0.01:
                omega_0_new = np.clip(abs(omega_0_new), min_vel, max_vel) * np.sign(omega_0_new)
            if abs(current_state.theta_1 - self.goal_state.theta_1) > 0.01:
                omega_1_new = np.clip(abs(omega_1_new), min_vel, max_vel) * np.sign(omega_1_new)
            
            # RK4 integration for position update
            k1_0 = current_state.omega_0
            k1_1 = current_state.omega_1
            k2_0 = omega_0_new
            k2_1 = omega_1_new
            
            theta_0_new = current_state.theta_0 + dt * (k1_0 + k2_0) / 2
            theta_1_new = current_state.theta_1 + dt * (k1_1 + k2_1) / 2
            
            # Discretize state
            theta_0_new = round(theta_0_new / resolution) * resolution
            theta_1_new = round(theta_1_new / resolution) * resolution
            
            if self.within_joint_limits((theta_0_new, theta_1_new)):
                neighbor = State(theta_0_new, theta_1_new, omega_0_new, omega_1_new)
                if self._check_kinematic_feasibility(current_state, neighbor):
                    neighbors.append(neighbor)
        
        return neighbors

    def planbid(self, start: Tuple[float, float], goal: Tuple[float, float], robot, 
                 final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
            """Plan a path from start to goal using bidirectional A* search."""
            # Reset planning state
            self.explored_states_count = 0
            self.best_distance_to_goal = float('inf')
            start_time = time.time()

            # Initialize forward search
            forward_start_state = State(robot.theta_0, robot.theta_1, robot.omega_0, robot.omega_1)
            goal_theta = robot.inverse_kinematics(*goal)
            forward_goal_state = State(goal_theta[0], goal_theta[1], final_velocities[0], final_velocities[1])
            self.goal_state = forward_goal_state

            # Initialize backward search
            backward_start_state = forward_goal_state
            backward_goal_state = forward_start_state

            # Initialize forward direction
            forward_open = [(self.heuristic(forward_start_state, forward_goal_state), 0, forward_start_state)]
            forward_closed: Set[State] = set()
            forward_came_from: Dict[State, Optional[State]] = {forward_start_state: None}
            forward_g_score: Dict[State, float] = {forward_start_state: 0}

            # Initialize backward direction
            backward_open = [(self.heuristic(backward_start_state, backward_goal_state), 0, backward_start_state)]
            backward_closed: Set[State] = set()
            backward_came_from: Dict[State, Optional[State]] = {backward_start_state: None}
            backward_g_score: Dict[State, float] = {backward_start_state: 0}

            # Track best solutions found
            best_path = None
            best_distance = float('inf')
            meeting_point = None

            while forward_open and backward_open:
                # Process forward search
                if forward_open:
                    current_forward = forward_open[0][2]
                    current_forward_pos = robot.forward_kinematics(current_forward.theta_0, current_forward.theta_1)
                    forward_distance = np.hypot(current_forward_pos[0] - goal[0], 
                                              current_forward_pos[1] - goal[1])

                    # Check if forward and backward searches meet
                    for state in backward_closed:
                        if self.states_close_enough(current_forward, state):
                            meeting_point = (current_forward, state)
                            break

                    # Update best path if this one is closer
                    if forward_distance < best_distance:
                        best_distance = forward_distance
                        best_path = self.reconstruct_bidirectional_path(
                            current_forward, forward_came_from, backward_came_from, meeting_point)
                        self.last_progress_time = time.time()

                    _, _, current = heapq.heappop(forward_open)
                    forward_closed.add(current)

                    # Expand forward neighbors
                    for neighbor in self.get_neighbors(current):
                        if neighbor in forward_closed:
                            continue

                        tentative_g_score = forward_g_score[current] + self.distance(current, neighbor)

                        if neighbor not in forward_g_score or tentative_g_score < forward_g_score[neighbor]:
                            if not self.is_collision(neighbor):
                                forward_came_from[neighbor] = current
                                forward_g_score[neighbor] = tentative_g_score
                                f_score = tentative_g_score + self.heuristic(neighbor, forward_goal_state)
                                heapq.heappush(forward_open, (f_score, self.explored_states_count, neighbor))

                # Process backward search
                if backward_open:
                    current_backward = backward_open[0][2]
                    
                    # Check if forward and backward searches meet
                    for state in forward_closed:
                        if self.states_close_enough(current_backward, state):
                            meeting_point = (state, current_backward)
                            break

                    _, _, current = heapq.heappop(backward_open)
                    backward_closed.add(current)

                    # Expand backward neighbors
                    for neighbor in self.get_neighbors(current):
                        if neighbor in backward_closed:
                            continue

                        tentative_g_score = backward_g_score[current] + self.distance(current, neighbor)

                        if neighbor not in backward_g_score or tentative_g_score < backward_g_score[neighbor]:
                            if not self.is_collision(neighbor):
                                backward_came_from[neighbor] = current
                                backward_g_score[neighbor] = tentative_g_score
                                f_score = tentative_g_score + self.heuristic(neighbor, backward_goal_state)
                                heapq.heappush(backward_open, (f_score, self.explored_states_count, neighbor))

                # Check timeout and progress
                if time.time() - start_time > self.planning_timeout:
                    self.debug_helper.log_state("Planning timeout - returning best path")
                    return best_path if best_path else self.reconstruct_bidirectional_path(
                        current_forward, forward_came_from, backward_came_from, meeting_point)

                if time.time() - self.last_progress_time > 5.0 and best_path:
                    self.debug_helper.log_state("No progress - returning best path")
                    return best_path

                # Meeting point found
                if meeting_point:
                    return self.reconstruct_bidirectional_path(
                        meeting_point[0], forward_came_from, backward_came_from, meeting_point)

                # Memory management
                if self.explored_states_count % 1000 == 0:
                    self.clear_cache()

                self.explored_states_count += 1
                if self.explored_states_count > self.max_explored_states:
                    return best_path if best_path else self.reconstruct_bidirectional_path(
                        current_forward, forward_came_from, backward_came_from, meeting_point)

            raise ValueError("No path found")

    def states_close_enough(self, state1: State, state2: State, threshold: float = 0.1) -> bool:
        """Check if two states are kinodynamically connectable"""
        pos1 = self.robot.forward_kinematics(state1.theta_0, state1.theta_1)
        pos2 = self.robot.forward_kinematics(state2.theta_0, state2.theta_1)
        
        # Check position difference
        pos_diff = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        
        # Check velocity compatibility
        vel_diff = np.hypot(state1.omega_0 - state2.omega_0, state1.omega_1 - state2.omega_1)
        
        # Check if connection respects acceleration limits
        dt = self.constants.DT
        required_accel_0 = abs(state2.omega_0 - state1.omega_0) / dt
        required_accel_1 = abs(state2.omega_1 - state1.omega_1) / dt
        
        accel_feasible = (required_accel_0 <= self.constants.MAX_ACCELERATION and 
                          required_accel_1 <= self.constants.MAX_ACCELERATION)
        
        return pos_diff < threshold and vel_diff < self.constants.VELOCITY_RESOLUTION and accel_feasible

    def interpolate_states(self, state1: State, state2: State, num_steps: int = 10) -> List[State]:
        """Generate a kinodynamically feasible interpolation between states"""
        states = []
        dt = self.constants.DT
        
        # Calculate required accelerations
        delta_omega_0 = state2.omega_0 - state1.omega_0
        delta_omega_1 = state2.omega_1 - state1.omega_1
        
        alpha_0 = delta_omega_0 / (num_steps * dt)
        alpha_1 = delta_omega_1 / (num_steps * dt)
        
        # Generate intermediate states
        for i in range(num_steps + 1):
            t = i * dt
            
            # Update velocities with constant acceleration
            omega_0 = state1.omega_0 + alpha_0 * t
            omega_1 = state1.omega_1 + alpha_1 * t
            
            # Update positions considering velocities
            theta_0 = state1.theta_0 + state1.omega_0 * t + 0.5 * alpha_0 * t * t
            theta_1 = state1.theta_1 + state1.omega_1 * t + 0.5 * alpha_1 * t * t
            
            states.append(State(theta_0, theta_1, omega_0, omega_1))
        
        return states

    def reconstruct_bidirectional_path(self, meeting_state: State, 
                                     forward_came_from: Dict[State, Optional[State]],
                                     backward_came_from: Dict[State, Optional[State]],
                                     meeting_point: Optional[Tuple[State, State]] = None) -> List[State]:
        """Reconstruct the path with kinodynamic interpolation at meeting point"""
        forward_path = []
        current = meeting_point[0] if meeting_point else meeting_state
        
        # Reconstruct forward path
        while current in forward_came_from:
            forward_path.append(current)
            current = forward_came_from[current]
            if current is None:
                break
        
        # Reconstruct backward path
        backward_path = []
        current = meeting_point[1] if meeting_point else meeting_state
        while current in backward_came_from:
            backward_path.append(current)
            current = backward_came_from[current]
            if current is None:
                break
                
        # If we have a meeting point, create a smooth connection
        if meeting_point:
            connecting_states = self.interpolate_states(meeting_point[0], meeting_point[1])
            return forward_path[::-1] + connecting_states + backward_path
        else:
            # Just reverse and combine paths if no meeting point
            return forward_path[::-1] + backward_path

    def Plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot, 
                    final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """Plan a path from start to goal with trajectory optimization."""
        # Get initial path
        try:
            # First try coarse planning
            self.set_mode('kinodynamic')
            path = self.plan(start, goal, robot, final_velocities)
            self.debug_helper.print_path_stats(path, robot)
            self.debug_helper.validate_path_limits(path, robot)
            self.debug_helper.print_path_points(path)
            return path
            coarse_path = self.plan(start, goal, robot)
            if not coarse_path:
                self.debug_helper.log_state("Coarse planning failed, attempting kinodynamic planning")
                self.set_mode('kinodynamic')
                path = plan(start, goal, robot, final_velocities)
                self.debug_helper.print_path_stats(path, robot)
                self.debug_helper.validate_path_limits(path, robot)
                self.debug_helper.print_path_points(path)
                return path
                
            # If coarse path found, refine it with kinodynamic constraints
            self.set_mode('kinodynamic')
            refined_path = self._refine_path(coarse_path, robot)
            coarse_path_last_state = coarse_path[-1] 
            goal_pos = robot.forward_kinematics(goal[0], goal[1])
            coarse_path_last_pos = robot.forward_kinematics(coarse_path_last_state.theta_0, coarse_path_last_state.theta_1)
            dist_to_goal = np.hypot(goal_pos[0] - coarse_path_last_pos[0], goal_pos[1] - coarse_path_last_pos[1])

            self.debug_helper.log_state(f"dist_to_goal: {dist_to_goal:.3f}")
            
            if self.best_distance_to_goal < 2.0:
                self.debug_helper.log_state("Successfully refined coarse path")
                self.debug_helper.print_path_stats(refined_path, robot)
                self.debug_helper.validate_path_limits(refined_path, robot)
                self.debug_helper.print_path_points(refined_path)
                return refined_path
            else:
                self.debug_helper.log_state("Coarse path refining fails, attempting kinodynamic planning")
                path_segments = 5
                for i in range(path_segments,0 , -1):
                    self.debug_helper.log_state("Attempting to connect coarse path to goal")
                    # Get state from coarse path
                    if refined_path:
                        intermediate_state = refined_path[(int(len(coarse_path) * (i) / path_segments)) - 1]
                    else:
                        intermediate_state = coarse_path[(int(len(coarse_path) * (i) / path_segments)) - 1]
                    # Extract state information properly
                    new_goal = (intermediate_state.theta_0, intermediate_state.theta_1)
                    # Use velocities from the state if available
                    velocities = (getattr(intermediate_state, 'velocity_0', 0.0),
                                getattr(intermediate_state, 'velocity_1', 0.0))
                    
                    path = self.plan(goal, new_goal, robot, velocities)
                    
                    if path:
                        path.reverse()
                        path_final = []
                        if refined_path:
                            # Use proper indexing for refined path
                            split_index = (int(len(refined_path) * i / path_segments)) - 1
                            path_final = refined_path[:split_index]
                            path_final.extend(path)
                        else:
                            # Use proper indexing for coarse path
                            split_index = (int(len(coarse_path) * i / path_segments)) - 1
                            path_final = coarse_path[:split_index]
                            path_final.extend(path)
                        
                        self.debug_helper.print_path_stats(path_final, robot)
                        self.debug_helper.print_path_points(path_final)
                        return path_final
                        
        except Exception as e:
            self.debug_helper.log_state(f"Planning error: {str(e)}")
            return []

    def _refine_path(self, coarse_path: List[State], robot: Robot) -> List[State]:
        """Refine coarse path with kinodynamic constraints and smoothness"""
        if not coarse_path:
            return None

        try:
            # Initial refinement with velocity constraints
            refined_path = self._initial_velocity_refinement(coarse_path)
            if not refined_path:
                return None

            # Apply acceleration and jerk constraints
            smoothed_path = self._apply_smoothness_constraints(refined_path)
            if not smoothed_path:
                return None

            # Final velocity profile optimization
            optimized_path = self._optimize_velocity_profile(smoothed_path)
            
            return optimized_path

        except Exception as e:
            self.debug_helper.log_state(f"Path refinement error: {str(e)}")
            return None

    def _initial_velocity_refinement(self, coarse_path: List[State]) -> List[State]:
        """Initial refinement with independent joint constraints"""
        refined_path = []
        dt = self.constants.DT

        # Add start state
        refined_path.append(coarse_path[0])

        for i in range(1, len(coarse_path)):
            prev_state = refined_path[-1]
            target_state = coarse_path[i]

            # Compute required velocities for each joint independently
            delta_theta_0 = target_state.theta_0 - prev_state.theta_0
            delta_theta_1 = target_state.theta_1 - prev_state.theta_1

            # Compute independent time scales for each joint based on velocity limits
            time_scale_0 = max(1.0, abs(delta_theta_0) / (self.constants.MAX_VELOCITY * dt))
            time_scale_1 = max(1.0, abs(delta_theta_1) / (self.constants.MAX_VELOCITY * dt))
            
            # Use the longer time scale to ensure both joints respect limits
            time_scale = max(time_scale_0, time_scale_1)
            
            # Compute velocities independently
            omega_0 = delta_theta_0 / (dt * time_scale)
            omega_1 = delta_theta_1 / (dt * time_scale)

            # Enforce velocity limits independently for each joint
            omega_0 = np.clip(omega_0, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)
            omega_1 = np.clip(omega_1, -self.constants.MAX_VELOCITY, self.constants.MAX_VELOCITY)

            # Create intermediate states with independent joint consideration
            num_steps = max(1, int(time_scale))
            for step in range(num_steps):
                alpha = (step + 1) / num_steps
                
                # Independent position interpolation
                theta_0 = prev_state.theta_0 + alpha * delta_theta_0
                theta_1 = prev_state.theta_1 + alpha * delta_theta_1
                
                # Independent velocity profiles using bell curve for each joint
                bell_shape = 1.0 - 0.5 * abs(alpha - 0.5)  # Bell curve factor
                current_omega_0 = omega_0 * bell_shape if abs(delta_theta_0) > 1e-6 else 0.0
                current_omega_1 = omega_1 * bell_shape if abs(delta_theta_1) > 1e-6 else 0.0

                # Create state and check independent joint constraints
                refined_state = State(theta_0, theta_1, current_omega_0, current_omega_1)
                
                # Verify kinematic feasibility considering joints independently
                if self._check_kinematic_feasibility(prev_state, refined_state):
                    refined_path.append(refined_state)
                else:
                    # Try scaling down velocities independently
                    scale_factor = 0.8  # Reduce velocities by 20%
                    adjusted_state = State(
                        theta_0, 
                        theta_1,
                        current_omega_0 * scale_factor if abs(current_omega_0) > self.constants.MAX_VELOCITY else current_omega_0,
                        current_omega_1 * scale_factor if abs(current_omega_1) > self.constants.MAX_VELOCITY else current_omega_1
                    )
                    
                    if self._check_kinematic_feasibility(prev_state, adjusted_state):
                        refined_path.append(adjusted_state)
                    else:
                        return None

        return refined_path

    def _apply_smoothness_constraints(self, path: List[State]) -> List[State]:
        """Apply acceleration and jerk constraints with independent joint handling"""
        if len(path) < 3:
            return path

        smoothed_path = []
        dt = self.constants.DT
        window = self.smoothing_window

        # Add initial states
        smoothed_path.extend(path[:2])

        for i in range(2, len(path) - 1):
            # Get window of states
            start_idx = max(0, i - window)
            end_idx = min(len(path), i + window + 1)
            window_states = path[start_idx:end_idx]

            # Current state and its neighbors
            prev_state = smoothed_path[-1]
            current_state = path[i]
            next_state = path[i + 1]

            # Compute accelerations for each joint independently
            current_acc_0 = (current_state.omega_0 - prev_state.omega_0) / dt
            current_acc_1 = (current_state.omega_1 - prev_state.omega_1) / dt
            next_acc_0 = (next_state.omega_0 - current_state.omega_0) / dt
            next_acc_1 = (next_state.omega_1 - current_state.omega_1) / dt

            # Compute jerks for each joint independently
            jerk_0 = (next_acc_0 - current_acc_0) / dt
            jerk_1 = (next_acc_1 - current_acc_1) / dt

            # Apply independent jerk limits and compute resulting states
            next_state = self._apply_independent_jerk_limits(
                current_state, next_state, 
                current_acc_0, current_acc_1,
                jerk_0, jerk_1, dt
            )

            # Smooth velocities independently using window average
            if len(window_states) >= 3:
                smoothed_omega_0 = self._smooth_velocity_independent(
                    [s.omega_0 for s in window_states],
                    current_state.omega_0,
                    self.constants.MAX_ACCELERATION
                )
                smoothed_omega_1 = self._smooth_velocity_independent(
                    [s.omega_1 for s in window_states],
                    current_state.omega_1,
                    self.constants.MAX_ACCELERATION
                )

                smoothed_state = State(
                    current_state.theta_0,
                    current_state.theta_1,
                    smoothed_omega_0,
                    smoothed_omega_1
                )

                if self._check_independent_kinematic_feasibility(prev_state, smoothed_state):
                    smoothed_path.append(smoothed_state)
                else:
                    # Try independent scaling of velocities
                    scaled_state = self._scale_velocities_independently(prev_state, smoothed_state)
                    if scaled_state:
                        smoothed_path.append(scaled_state)
                    else:
                        smoothed_path.append(current_state)
            else:
                smoothed_path.append(current_state)

        # Add final state
        smoothed_path.append(path[-1])
        return smoothed_path

    def _apply_independent_jerk_limits(self, current_state: State, next_state: State,
                                     current_acc_0: float, current_acc_1: float,
                                     jerk_0: float, jerk_1: float, dt: float) -> State:
        """Apply jerk limits independently to each joint"""
        # Handle joint 0
        if abs(jerk_0) > self.max_jerk:
            scale_0 = self.max_jerk / abs(jerk_0)
            next_acc_0 = current_acc_0 + jerk_0 * scale_0 * dt
            omega_0 = current_state.omega_0 + next_acc_0 * dt
        else:
            omega_0 = next_state.omega_0

        # Handle joint 1 independently
        if abs(jerk_1) > self.max_jerk:
            scale_1 = self.max_jerk / abs(jerk_1)
            next_acc_1 = current_acc_1 + jerk_1 * scale_1 * dt
            omega_1 = current_state.omega_1 + next_acc_1 * dt
        else:
            omega_1 = next_state.omega_1

        return State(
            next_state.theta_0,
            next_state.theta_1,
            omega_0,
            omega_1
        )

    def _smooth_velocity_independent(self, velocities: List[float], current_velocity: float, 
                                   max_accel: float, min_velocity: float = 0.01) -> float:
        """Smooth velocity independently for each joint with improved constraints"""
        if not velocities:
            return current_velocity

        # Compute weighted average with more weight to nearby points
        weights = [1.0 - abs(i - len(velocities)//2)/(len(velocities)//2) for i in range(len(velocities))]
        weighted_avg = np.average(velocities, weights=weights)

        # Limit change based on acceleration constraint
        dt = self.constants.DT
        max_change = max_accel * dt
        delta_v = weighted_avg - current_velocity

        # Apply acceleration limits
        if abs(delta_v) > max_change:
            delta_v = np.sign(delta_v) * max_change

        # Ensure minimum velocity if moving
        new_velocity = current_velocity + delta_v
        if abs(new_velocity) < min_velocity and abs(current_velocity) > min_velocity:
            new_velocity = np.sign(current_velocity) * min_velocity

        return new_velocity

    def _check_independent_kinematic_feasibility(self, current: State, next: State) -> bool:
        """Check kinematic feasibility with independent joint constraints"""
        dt = self.constants.DT
        max_accel = self.constants.MAX_ACCELERATION
        max_vel = self.constants.MAX_VELOCITY

        # Check velocity limits independently
        if (abs(next.omega_0) > max_vel or abs(next.omega_1) > max_vel):
            return False

        # Compute accelerations for each joint
        alpha_0 = (next.omega_0 - current.omega_0) / dt
        alpha_1 = (next.omega_1 - current.omega_1) / dt

        # Check acceleration limits independently
        accel_tolerance = 1.5  # Allow slightly higher accelerations
        max_allowed = max_accel * accel_tolerance

        if abs(alpha_0) > max_allowed or abs(alpha_1) > max_allowed:
            return False

        # Check velocity changes independently
        max_vel_change = max_vel * 0.5
        if (abs(next.omega_0 - current.omega_0) > max_vel_change or
            abs(next.omega_1 - current.omega_1) > max_vel_change):
            return False

        return True

    def _scale_velocities_independently(self, prev_state: State, state: State) -> Optional[State]:
        """Scale velocities independently when limits are exceeded"""
        dt = self.constants.DT
        max_vel = self.constants.MAX_VELOCITY
        max_accel = self.constants.MAX_ACCELERATION

        # Compute required accelerations
        alpha_0 = (state.omega_0 - prev_state.omega_0) / dt
        alpha_1 = (state.omega_1 - prev_state.omega_1) / dt

        # Compute scaling factors independently
        scale_0 = 1.0
        scale_1 = 1.0

        # Scale based on velocity limits
        if abs(state.omega_0) > max_vel:
            scale_0 = max_vel / abs(state.omega_0)
        if abs(state.omega_1) > max_vel:
            scale_1 = max_vel / abs(state.omega_1)

        # Scale based on acceleration limits
        if abs(alpha_0) > max_accel:
            scale_0 = min(scale_0, max_accel / abs(alpha_0))
        if abs(alpha_1) > max_accel:
            scale_1 = min(scale_1, max_accel / abs(alpha_1))

        # Apply independent scaling
        scaled_state = State(
            state.theta_0,
            state.theta_1,
            state.omega_0 * scale_0,
            state.omega_1 * scale_1
        )

        if self._check_independent_kinematic_feasibility(prev_state, scaled_state):
            return scaled_state
        return None

    def _optimize_velocity_profile(self, path: List[State]) -> List[State]:
        """Optimize velocity profile for smoothness"""
        if len(path) < 3:
            return path

        optimized_path = []
        dt = self.constants.DT

        # Add start state
        optimized_path.append(path[0])

        try:
            for i in range(1, len(path) - 1):
                prev_state = optimized_path[-1]
                current_state = path[i]
                next_state = path[i + 1]

                # Compute optimal velocities based on position and acceleration
                delta_pos_0 = next_state.theta_0 - prev_state.theta_0
                delta_pos_1 = next_state.theta_1 - prev_state.theta_1

                # Use trapezoidal velocity profile
                avg_vel_0 = delta_pos_0 / (2 * dt)
                avg_vel_1 = delta_pos_1 / (2 * dt)

                # Apply velocity limits with smooth transitions
                scale = min(1.0, self.constants.MAX_VELOCITY / (max(abs(avg_vel_0), abs(avg_vel_1)) + 1e-6))
                
                opt_omega_0 = avg_vel_0 * scale
                opt_omega_1 = avg_vel_1 * scale

                # Create optimized state
                optimized_state = State(
                    current_state.theta_0,
                    current_state.theta_1,
                    opt_omega_0,
                    opt_omega_1
                )

                if self._check_kinematic_feasibility(prev_state, optimized_state):
                    optimized_path.append(optimized_state)
                else:
                    optimized_path.append(current_state)

            # Add final state
            optimized_path.append(path[-1])
            return optimized_path

        except Exception as e:
            self.debug_helper.log_state(f"Path refinement error: {str(e)}")
            return None

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot, 
             final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """Internal planning method."""
        
        if not hasattr(self, 'debug_helper'):
            self.debug_helper = DebugHelper(debug=True)
            
        self.debug_helper.print_planning_header()
        self.debug_helper.log_state(f"Planning path from {start} to {goal}")
        
        # Reset planning state
        self.explored_states_count = 0
        self.best_distance_to_goal = float('inf')
        best_path = None
        
        try:
            # Get goal configuration
            goal_theta = robot.inverse_kinematics(*goal)
            start_theta = robot.inverse_kinematics(*start)
            
            # Create start and goal states
            if self.mode == 'kinodynamic':
                start_state = State(start_theta[0], start_theta[1], 0.0, 0.0)
                goal_state = State(goal_theta[0], goal_theta[1], final_velocities[0], final_velocities[1])
            else:
                start_state = State(start_theta[0], start_theta[1], 0.0, 0.0)
                goal_state = State(goal_theta[0], goal_theta[1], 0.0, 0.0)
            

            # Initialize planning
            self.goal_state = goal_state
            open_set = [(self.heuristic_coarse(start_state, goal_state) if self.mode == 'coarse' else self.heuristic(start_state, goal_state), 0, start_state)]
            closed_set = set()
            came_from = {start_state: None}
            g_score = {start_state: 0}
            max_no_neighbor_attempts = 20
            no_neighbor_count = 0;
            start_time = time.time()
            while open_set:
                current_state = open_set[0][2]
                try:
                    self.check_timeout_and_progress(open_set[0][2], goal_state, start_time)
                except PlanningTimeoutError as e:
                    # Return best path found so far
                    return self.reconstruct_path(came_from, open_set[0][2])
                
                current_pos = robot.forward_kinematics(current_state.theta_0, current_state.theta_1)
                current_distance = np.hypot(current_pos[0] - goal[0], current_pos[1] - goal[1])
                
                if current_distance < self.best_distance_to_goal:
                    self.best_distance_to_goal = current_distance
                    best_path = self.reconstruct_path(came_from, current_state)
                    self.debug_helper.log_state(f"New best distance: {current_distance:.3f}")
                    self.last_progress_time = time.time()
                    
                # Check if we're at goal
                if (self.is_goal_coarse(current_state, goal_state) if self.mode == 'coarse' else self.is_goal(current_state, goal_state)):
                    path = self.reconstruct_path(came_from, current_state)
                    if self.mode == 'coarse':
                        self.debug_helper.log_state("Path found by coarse Planner")
                        return path
                    else:
                         if self.debug_helper.validate_planner_output(path, start, goal, robot):
                            self.debug_helper.log_state("Path found by kinodynamic Planner")
                            return path
                
                _, _, current = heapq.heappop(open_set)
                closed_set.add(current)
                
                # Expand neighbors
                # Get neighbors based on mode
                neighbors = (self.get_neighbors_coarse(current) if self.mode == 'coarse' else self.get_neighbors(current))
                # Handle case where no valid neighbors are found
                if not neighbors:
                    no_neighbor_count += 1
                    self.debug_helper.log_state(f"No valid neighbors found at distance {current_distance:.3f}")
                    
                    if no_neighbor_count >= max_no_neighbor_attempts:
                        self.debug_helper.log_state(f"No valid neighbors found for {max_no_neighbor_attempts} consecutive attempts")
                        if best_path:
                            self.debug_helper.log_state("Returning best path found so far")
                            return best_path
                        else:
                            self.debug_helper.log_state("No valid path found")
                            return []
                    continue
                for neighbor in neighbors:
                    if neighbor in closed_set:
                        continue
                        
                    # Calculate cost based on mode
                    tentative_g_score = g_score[current] + (
                        self.distance_coarse(current, neighbor) if self.mode == 'coarse' else self.distance(current, neighbor)
                    )
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        if not self.is_collision(neighbor):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score = tentative_g_score + (
                                self.heuristic_coarse(neighbor, goal_state) if self.mode == 'coarse'else self.heuristic(neighbor, goal_state)
                            )
                            heapq.heappush(open_set, (f_score, self.explored_states_count, neighbor))
                
                if time.time() - self.last_progress_time > self.progress_timeout and best_path:
                    self.debug_helper.log_state("No progress - returning best path")
                    return best_path
                self.explored_states_count += 1
                if self.explored_states_count > self.max_explored_states:
                    self.debug_helper.log_state("Max explored states reached")
                    return best_path if best_path else []
                    
            # If we get here, no path was found
            self.debug_helper.log_state("No path was found")
            return best_path if best_path else []
            
        except Exception as e:
            self.debug_helper.log_state(f"Planning error: {str(e)}")
            return []

class SBPLLatticePlanner(GridBasedPlanner):
    """
    SBPL-style lattice planner using optimized motion primitives with acceleration constraints.
    """
    def __init__(self, robot, world):
        super().__init__(robot, world)
        self.velocity_resolution = self.constants.VELOCITY_RESOLUTION
        # Pre-compute and store motion primitives
        self.motion_primitives = self._generate_motion_primitives()
        # Initialize tracking variables
        self.explored_states_count = 0
        self.best_distance_to_goal = float('inf')
        self.last_progress_time = 0
        # Add adaptive primitive selection
        self.successful_primitives = set()
        self.primitive_success_count = {}

    def _generate_motion_primitives(self) -> List[Tuple[float, float]]:
        """Generate optimized set of motion primitives with fewer samples"""
        max_alpha = self.constants.MAX_ACCELERATION
        # Use fewer acceleration levels
        num_accels = 5  # Reduced from 9
        accelerations = np.linspace(-max_alpha, max_alpha, num_accels)
        primitives = []
        for alpha_0 in accelerations:
            for alpha_1 in accelerations:
                primitives.append((alpha_0, alpha_1))
                # Initialize success count
                self.primitive_success_count[(alpha_0, alpha_1)] = 0
        return primitives

    def update_primitive_success(self, primitive: Tuple[float, float], success: bool):
        """Update the success count for a motion primitive"""
        if success:
            self.primitive_success_count[primitive] += 1
            self.successful_primitives.add(primitive)

    def get_neighbors(self, current_state: State) -> List[State]:
        """Get neighbors using adaptive motion primitive selection"""
        neighbors = []
        dt = self.constants.DT
        max_vel = self.constants.MAX_VELOCITY

        # Use successful primitives more frequently
        primitives_to_try = list(self.successful_primitives) if self.successful_primitives else self.motion_primitives
        
        for alpha_0, alpha_1 in primitives_to_try:
            # Compute new velocities with acceleration constraints
            omega_0_new = np.clip(
                current_state.omega_0 + alpha_0 * dt,
                -max_vel, max_vel
            )
            omega_1_new = np.clip(
                current_state.omega_1 + alpha_1 * dt,
                -max_vel, max_vel
            )

            # Compute new positions
            theta_0_new = current_state.theta_0 + omega_0_new * dt
            theta_1_new = current_state.theta_1 + omega_1_new * dt

            # Discretize state
            theta_0_new, theta_1_new = self.discretize_state((theta_0_new, theta_1_new))
            omega_0_new = round(omega_0_new / self.velocity_resolution) * self.velocity_resolution
            omega_1_new = round(omega_1_new / self.velocity_resolution) * self.velocity_resolution

            if self.within_joint_limits((theta_0_new, theta_1_new)):
                neighbor = State(theta_0_new, theta_1_new, omega_0_new, omega_1_new)
                neighbors.append(neighbor)
                # Update primitive success
                if not self.is_collision(neighbor):
                    self.update_primitive_success((alpha_0, alpha_1), True)

        return neighbors

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], robot, 
            final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        # Reset tracking variables
        self.explored_states_count = 0
        self.best_distance_to_goal = float('inf')
        start_time = time.time()
        
        # Initialize states
        start_theta = self.discretize_state(robot.inverse_kinematics(*start))
        goal_theta = self.discretize_state(robot.inverse_kinematics(*goal))
        start_state = State(start_theta[0], start_theta[1], 0.0, 0.0)
        goal_state = State(goal_theta[0], goal_theta[1], final_velocities[0], final_velocities[1])

        # Priority queue for open set
        open_set = [(self.heuristic(start_state, goal_state), 0, start_state)]
        closed_set: Set[State] = set()
        came_from: Dict[State, Optional[State]] = {start_state: None}
        g_score: Dict[State, float] = {start_state: 0}

        while open_set:
            try:
                self.check_timeout_and_progress(open_set[0][2], goal_state, start_time)
            except PlanningTimeoutError as e:
                # Return best path found so far
                return self.reconstruct_path(came_from, open_set[0][2])

            _, _, current = heapq.heappop(open_set)

            if self.is_goal(current, goal_state):
                return self.reconstruct_path(came_from, current)

            self.explored_states_count += 1
            if self.explored_states_count > self.max_explored_states:
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
                        f_score = tentative_g_score + self.heuristic(neighbor, goal_state)
                        heapq.heappush(open_set, (f_score, self.explored_states_count, neighbor))

            # Periodically clear cache
            if self.explored_states_count % 1000 == 0:
                self.clear_cache()

        raise ValueError("No path found")

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

class MPCController:
    """Model Predictive Controller for 2-link robotic arm"""
    def __init__(self, robot, constants):
        self.robot = robot
        self.constants = constants
        self.h = 0.1716  # Prediction horizon
        self.dt = constants.DT
        self.zeta = 0.9  # Damping ratio
        self.w0 = 4.0  # Natural frequency - should be tuned
        self.path = []  # Store reference path
        self.path_index = 0  # Current path index

        self.debug_helper = DebugHelper(debug=False)
        
        # Compute controller gains based on provided equations
        self.k1 = 2 / (self.h**2 + 4*self.get_rho())
        self.k2 = (2*self.h**2 + 4*self.get_rho()) / (self.h**3 + 4*self.get_rho()*self.h)

        self.debug_helper.log_state(f"\nMPC Controller Initialized:")
        self.debug_helper.log_state(f"Horizon: {self.h}, DT: {self.dt}")
        self.debug_helper.log_state(f"Controller gains - k1: {self.k1:.3f}, k2: {self.k2:.3f}")

        
        # Initialize state histories
        self.theta_ref_history = []
        self.theta_actual_history = []
        self.v_history = []
        self.tau_history = []

    def get_rho(self) -> float:
        """Compute weight factor ρ based on given equation"""
        w0h = self.w0 * self.h
        rho = (2 - w0h**2) / (4 * self.w0**2)
        self.debug_helper.log_state(f"Computed rho: {rho:.3f}")
        return rho

    def compute_feedback_linearization(self, theta, theta_dot, v):
        self.debug_helper.log_state("\nComputing Feedback Linearization:")
        self.debug_helper.log_state(f"Current state - theta: {theta}, theta_dot: {theta_dot}")
        self.debug_helper.log_state(f"Synthetic control input v: {v}")
        
        M, C, G = self.robot.compute_dynamics(theta[0], theta[1], theta_dot[0], theta_dot[1])
        
        self.debug_helper.log_state("\nDynamics matrices:")
        self.debug_helper.log_state(f"M:\n{M}")
        self.debug_helper.log_state(f"C:\n{C}")
        self.debug_helper.log_state(f"G:\n{G}")
        
        tau = M @ np.array(v) + C @ np.array(theta_dot) + G
        
        self.debug_helper.log_state(f"Computed torques: {tau}")
        return tau[0], tau[1]
    def compute_synthetic_control(self, theta_ref, theta, theta_dot):
        self.debug_helper.log_state("\nComputing Synthetic Control:")
        self.debug_helper.log_state(f"Reference theta: {theta_ref}")
        self.debug_helper.log_state(f"Current theta: {theta}")
        self.debug_helper.log_state(f"Current theta_dot: {theta_dot}")
        
        v1 = self.k1 * (theta_ref[0] - theta[0]) - self.k2 * theta_dot[0]
        v2 = self.k1 * (theta_ref[1] - theta[1]) - self.k2 * theta_dot[1]
        
        self.debug_helper.log_state(f"Position errors: ({theta_ref[0] - theta[0]:.3f}, {theta_ref[1] - theta[1]:.3f})")
        self.debug_helper.log_state(f"Computed synthetic controls: v1={v1:.3f}, v2={v2:.3f}")
        
        return v1, v2

    def predict_trajectory(self, current_state: Tuple[float, float, float, float],
                         control_sequence: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Predict system trajectory over horizon"""
        trajectory = []
        state = list(current_state)
        
        for v in control_sequence:
            # Simulate system dynamics
            tau = self.compute_feedback_linearization(
                (state[0], state[1]),
                (state[2], state[3]),
                v
            )
            
            # Update state using robot dynamics
            alpha_0, alpha_1 = self.robot.forward_dynamics(
                state[0], state[1],
                state[2], state[3],
                tau[0], tau[1]
            )
            
            # Euler integration
            state[2] += alpha_0 * self.dt  # Update velocities
            state[3] += alpha_1 * self.dt
            state[0] += state[2] * self.dt  # Update positions
            state[1] += state[3] * self.dt
            
            trajectory.append((state[0], state[1]))
            
        return trajectory

    def optimize_control_sequence(self, current_state: Tuple[float, float, float, float],
                                reference_trajectory: List[Tuple[float, float]], steps_in_horizon: int) -> List[Tuple[float, float]]:
        """Optimize control sequence over prediction horizon"""
        # Initialize control sequence
        control_sequence = []
        
        """Optimize control sequence with debugging"""
        self.debug_helper.log_state("\nOptimizing Control Sequence:")
        self.debug_helper.log_state(f"Current state: {current_state}")
        self.debug_helper.log_state(f"Reference trajectory length: {len(reference_trajectory)}")
        
        control_sequence = []
        position_threshold = 0.01
        
        for i in range(1, steps_in_horizon):
            self.debug_helper.log_state(f"\nOptimization step {i}:")
            
            ref_i = reference_trajectory[min(i, len(reference_trajectory) - 1)]
            if i < len(reference_trajectory) and (abs(current_state[0] - ref_i[0]) > position_threshold or 
                abs(current_state[1] - ref_i [1]) > position_threshold):
                self.debug_helper.log_state(f"Reference point: {ref_i}")
            
            # Compute control for this step
            v = self.compute_synthetic_control(
                ref_i,
                (current_state[0], current_state[1]),
                (current_state[2], current_state[3])
            )
            
            control_sequence.append(v)
            self.debug_helper.log_state(f"Computed control: {v}")
        
        return control_sequence

    def step(self, theta_ref: Tuple[float, float], robot) -> Tuple[float, float]:
        """Execute one control step"""
        self.debug_helper.log_state("\n=== MPC Step Execution ===")

        # Current state
        current_state = (robot.theta_0, robot.theta_1, robot.omega_0, robot.omega_1)
        self.debug_helper.log_state(f"Current state: θ0={robot.theta_0:.3f}, θ1={robot.theta_1:.3f}, ω0={robot.omega_0:.3f}, ω1={robot.omega_1:.3f}")
        self.debug_helper.log_state(f"Reference state: θ0={theta_ref[0]:.3f}, θ1={theta_ref[1]:.3f}")
        
        # Generate reference trajectory over time horizon
        reference_trajectory = []
        current_path_index = self.get_current_path_index()
        
        # Calculate how many path indices correspond to the time horizon
        # h is time horizon in seconds, dt is timestep, so h/dt gives number of steps
        steps_in_horizon = int(self.h / self.dt)

        self.debug_helper.log_state(f"\nGenerating reference trajectory:")
        self.debug_helper.log_state(f"Current path index: {current_path_index}")
        self.debug_helper.log_state(f"Steps in horizon: {steps_in_horizon}")
        self.debug_helper.log_state(f"horizon(s): {self.h}")
        self.debug_helper.log_state(f"Path length: {len(self.path)}")
        
        # Fill reference trajectory with bounds checking
        for i in range(1, steps_in_horizon):
            path_idx = min(current_path_index + i, len(self.path) - 1)
            
            if path_idx < len(self.path):
                ref_state = self.path[path_idx]
                reference_trajectory.append((ref_state.theta_0, ref_state.theta_1))
                self.debug_helper.log_state(f"Reference point {i}: θ0={ref_state.theta_0:.3f}, θ1={ref_state.theta_1:.3f}")
        # Optimize control sequence
        control_sequence = self.optimize_control_sequence(current_state, reference_trajectory, steps_in_horizon)


         # Select first non-zero control action
        # Instead of always taking first control, find first non-zero control
        selected_control = None
        min_control = 0.09  # Minimum control action
        for v in control_sequence:
            if abs(v[0]) > min_control or abs(v[1]) > min_control:
                selected_control = v
                break
        
        # # If all controls are zero, use the control for the next reference point
        # if not selected_control and len(control_sequence) > 1:
        #     selected_control = control_sequence[1]
        # else:
        #     selected_control = control_sequence[0]

        # FIX 2: Add minimum control threshold
        # Take first control action
        v = selected_control
        self.debug_helper.log_state(f"\nSelected control action: v={v}")
        
        # Compute torques
        tau = self.compute_feedback_linearization(
            (robot.theta_0, robot.theta_1),
            (robot.omega_0, robot.omega_1),
            v
        )
        
        self.debug_helper.log_state(f"Final computed torques: τ0={tau[0]:.3f}, τ1={tau[1]:.3f}")
        
        
        # Store history
        self.theta_ref_history.append(theta_ref)
        self.theta_actual_history.append((robot.theta_0, robot.theta_1))
        self.v_history.append(v)
        self.tau_history.append(tau)
        
        return tau

    def reset(self):
        """Reset controller state"""
        self.theta_ref_history = []
        self.theta_actual_history = []
        self.v_history = []
        self.tau_history = []
        self.path_index = 0

    def set_path(self, path: List[State]):
        """Set the reference path for the controller"""
        self.path = path
        self.path_index = 0

    def set_path_index(self, path_index: int):
        self.path_index = path_index

    def get_current_path_index(self) -> int:
        """Get current path index with bounds checking"""
        return min(self.path_index, len(self.path) - 1) if self.path else 0

    def update_path_index(self, robot):
        """Update path index based on tracking progress"""
        if not self.path:
            return

        current_pos = robot.joint_2_pos()
        
        # Look ahead window
        look_ahead = min(10, len(self.path) - self.path_index)
        min_dist = float('inf')
        best_idx = self.path_index
        
        # Find closest point on path within look-ahead window
        for i in range(self.path_index, self.path_index + look_ahead):
            if i >= len(self.path):
                break
                
            path_pos = robot.forward_kinematics(
                self.path[i].theta_0,
                self.path[i].theta_1
            )
            
            dist = np.hypot(current_pos[0] - path_pos[0],
                          current_pos[1] - path_pos[1])
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # Update index if making forward progress
        if best_idx > self.path_index:
            self.path_index = best_idx



class Controller:
    def __init__(self, robot: Robot, world: World, planner: Planner) -> None:
        self.constants = robot.constants
        self.world = world
        self.planner = planner
        self.path: List[State] = []
        self.path_index = 0
        # Initialize MPC controller
        self.mpc = MPCController(None, robot.constants)  # Robot will be set in set_goal
        
        
        # Initialize trajectory tracking variables
        self.reference_theta_0: List[float] = []
        self.reference_theta_1: List[float] = []
        self.reference_omega_0: List[float] = []
        self.reference_omega_1: List[float] = []
        self.reference_alpha_0: List[float] = []
        self.reference_alpha_1: List[float] = []
        
        # Initialize actual value tracking
        self.actual_theta_0: List[float] = []
        self.actual_theta_1: List[float] = []
        self.actual_omega_0: List[float] = []
        self.actual_omega_1: List[float] = []
        self.actual_alpha_0: List[float] = []
        self.actual_alpha_1: List[float] = []
        
        
        # Add debug helper
        self.debug_helper = DebugHelper(debug=True)

    def set_goal(self, robot: Robot, goal: Tuple[float, float], final_velocities: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Set a new goal and plan path"""
        self.debug_helper.print_controller_header()
        self.debug_helper.log_state(f"Setting new goal: {goal}, final velocities: {final_velocities}")
        
        start_pos = robot.joint_2_pos()
        self.goal = goal

        # Update robot reference in MPC controller
        self.mpc.robot = robot
        
        # Get path from planner
        try:
            coarse_path = self.planner.Plan(start_pos, goal, robot, final_velocities)
            
            # Generate trajectory from path
            # Initialize the planner
            trajectory_planner = TimeOptimalTrajectoryGenerator(robot, coarse_path)
            
            # Generate optimal trajectory
            path = trajectory_planner.generate_trajectory(coarse_path)
            self.debug_helper.print_trajectory_stats(path, robot)
            self.debug_helper.validate_trajectory_dynamics(path, robot)
            self.debug_helper.print_trajectory_points(path, robot)
            
            if not path:
                raise ValueError("No valid path found")

            # # Validate path continuity
            # for i in range(1, len(path)):
            #     delta_theta0 = abs(path[i].theta_0 - path[i-1].theta_0)
            #     delta_theta1 = abs(path[i].theta_1 - path[i-1].theta_1)
            #     if delta_theta0 > 0.1 or delta_theta1 > 0.1:
            #         self.debug_helper.log_state(f"Warning: Large path discontinuity at index {i}")
            
            # # Extract reference trajectories with validation
            # self.reference_theta_0 = []
            # self.reference_theta_1 = []
            # self.reference_omega_0 = []
            # self.reference_omega_1 = []
            # self.reference_alpha_0: List[float] = []
            # self.reference_alpha_1: List[float] = []

            # dt = self.constants.DT
            # for i in range(1, len(path)):
            #     if (abs(path[i].theta_0) <= self.constants.JOINT_LIMITS[1] and
            #         abs(path[i].theta_1) <= self.constants.JOINT_LIMITS[1]):
            #         self.reference_theta_0.append(path[i].theta_0)
            #         self.reference_theta_1.append(path[i].theta_1)
            #         self.reference_omega_0.append(path[i].omega_0)
            #         self.reference_omega_1.append(path[i].omega_1)

            #         # Compute accelerations from velocity differences
            #         if i < len(path) - 1:
            #             alpha_0 = (path[i+1].omega_0 - path[i].omega_0) / dt
            #             alpha_1 = (path[i+1].omega_1 - path[i].omega_1) / dt
            #         else:
            #             alpha_0 = 0.0  # Zero acceleration for final point
            #             alpha_1 = 0.0
                    
            #         self.reference_alpha_0.append(alpha_0)
            #         self.reference_alpha_1.append(alpha_1)
            #     else:
            #         self.debug_helper.log_state(f"Skipping invalid state: {path[i]}")
                    
            self.path = path
            self.mpc.set_path(path)
            self.path_index = 0
            
            # Reset controller states
            self._reset_controller_states()
            
            self.debug_helper.log_state("Goal setting completed successfully")
            self.debug_helper.analyze_path_tracking(self, robot)
            
        except Exception as e:
            self.debug_helper.log_state(f"Error in goal setting: {str(e)}")
            raise

    def step(self, robot: Robot) -> Robot:
        """Execute one control step"""
        if not self.path or self.path_index >= len(self.path):
            return self._handle_empty_path(robot)
            
        try:
            # Get reference state from path
            ref_state = self.path[self.path_index]
            theta_ref = (ref_state.theta_0, ref_state.theta_1)
            
            # Compute control action using MPC
            tau_0, tau_1 = self.mpc.step(theta_ref, robot)
            
            # Apply torques and update robot state
            alpha_0, alpha_1 = robot.forward_dynamics(
                robot.theta_0, robot.theta_1,
                robot.omega_0, robot.omega_1,
                tau_0, tau_1
            )
            
            # Update velocities
            robot.omega_0 += alpha_0 * self.constants.DT
            robot.omega_1 += alpha_1 * self.constants.DT
            
            # Update positions
            robot.theta_0 += robot.omega_0 * self.constants.DT
            robot.theta_1 += robot.omega_1 * self.constants.DT

            
            # Store actual values and update path index
            self._update_tracking_history(robot)
            self._update_path_index(robot)
            
            # Periodic debugging analysis (every 100 steps)
            if len(self.actual_theta_0) % 1000 == 0:
                self._perform_debug_analysis(robot)
            
        except Exception as e:
            self.debug_helper.log_state(f"Controller error: {str(e)}")
            return self._handle_error(robot)
            
        return robot

    def _reset_controller_states(self) -> None:
        """Reset all controller states and tracking variables"""
        # Reset integral terms
        self.integral_error_0 = 0.0
        self.integral_error_1 = 0.0
        
        # Reset previous errors
        self.prev_theta_error_0 = 0.0
        self.prev_theta_error_1 = 0.0
        
        # Reset tracking history
        self.actual_theta_0 = []
        self.actual_theta_1 = []
        self.actual_omega_0 = []
        self.actual_omega_1 = []
        self.actual_alpha_0 = []
        self.actual_alpha_1 = []
        
        # Reset reference trajectories
        self.reference_theta_0 = []
        self.reference_theta_1 = []
        self.reference_omega_0 = []
        self.reference_omega_1 = []
        self.reference_alpha_0 = []
        self.reference_alpha_1 = []
        
        # Reset path tracking
        self.path_index = 0
        
        self.debug_helper.log_state("Controller states reset")

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _update_integral(self, integral: float, error: float, dt: float) -> float:
        """Update integral term with anti-windup"""
        self.integral_window.append(error)
        
        # Use moving average for integral
        if self.integral_window:
            integral = np.mean(self.integral_window) * dt
        
        # Use back-calculation anti-windup
        if abs(integral) > self.max_integral:
            gain = 0.5  # Back-calculation gain
            integral = integral - gain * (abs(integral) - self.max_integral) * np.sign(integral)

        integral_zone = 0.2  # Only integrate when error is small
        if abs(error) < integral_zone:
            integral = integral + error * dt
            
        # Implement sliding anti-windup
        if abs(integral) > self.max_integral:
            integral = integral * 0.95  # Gradual reduction
            
            return np.clip(integral, -self.max_integral, self.max_integral)
        
        # Apply anti-windup
        return np.clip(integral, -self.max_integral, self.max_integral)

    def _compute_adaptive_gain(self, base_gain: float, error: float) -> float:
        """Compute adaptive gain based on error"""
        error_scale = min(1.0, abs(error) / np.pi)  # Scale by error magnitude
        return base_gain * (1.0 + 0.5 * error_scale)  # Increase gain with error

    def _compute_velocity_limit(self, robot: Robot) -> float:
        """Compute velocity limit based on distance to goal"""
        current_pos = robot.joint_2_pos()
        dist_to_goal = np.hypot(current_pos[0] - self.goal[0],
                               current_pos[1] - self.goal[1])
        
        # Scale velocity limit with distance to goal
        base_limit = self.constants.MAX_VELOCITY
        scale = min(1.0, dist_to_goal / 0.5)  # Start slowing at 0.5 units from goal
        return base_limit * scale

    def _smooth_velocity_limit(self, velocity: float, limit: float) -> float:
        """Apply smooth velocity limiting with better transition"""
        if abs(velocity) <= limit:
            return velocity
        return limit * np.tanh(velocity / limit)

    def _check_joint_limits(self, theta_0: float, theta_1: float) -> bool:
        """Check if joint angles are within limits"""
        joint_limits = self.constants.JOINT_LIMITS
        return (joint_limits[0] <= theta_0 <= joint_limits[1] and
                joint_limits[0] <= theta_1 <= joint_limits[1])

    def compute_reference_state(self, robot: Robot) -> State:
        """Compute reference state with interpolation"""
        if self.path_index >= len(self.path):
            return self.path[-1]
            
        current_pos = robot.joint_2_pos()
        lookahead = self._compute_lookahead_distance(robot)
        
        # Find interpolation point
        for i in range(self.path_index, len(self.path)):
            path_pos = robot.forward_kinematics(
                self.path[i].theta_0,
                self.path[i].theta_1
            )
            dist = np.hypot(path_pos[0] - current_pos[0],
                          path_pos[1] - current_pos[1])
            
            if dist >= lookahead:
                if i > 0:
                    # Interpolate between points
                    prev = self.path[i-1]
                    curr = self.path[i]
                    alpha = (lookahead - dist) / dist
                    
                    return State(
                        prev.theta_0 + alpha * (curr.theta_0 - prev.theta_0),
                        prev.theta_1 + alpha * (curr.theta_1 - prev.theta_1),
                        prev.omega_0 + alpha * (curr.omega_0 - prev.omega_0),
                        prev.omega_1 + alpha * (curr.omega_1 - prev.omega_1)
                    )
                return self.path[i]
        
        return self.path[-1]

    def compute_reference_state(self, robot: Robot) -> State:
        """Compute reference state with improved interpolation"""
        if self.path_index >= len(self.path):
            return self.path[-1]
                
        current_pos = robot.joint_2_pos()
        lookahead = self._compute_lookahead_distance(robot)
        
        # Get current reference point
        current_ref = self.path[self.path_index]
        
        # Look ahead for target point
        look_ahead_window = 5  # Smaller window for smoother reference
        for i in range(self.path_index, min(self.path_index + look_ahead_window, len(self.path))):
            path_pos = robot.forward_kinematics(
                self.path[i].theta_0,
                self.path[i].theta_1
            )
            dist = np.hypot(path_pos[0] - current_pos[0],
                          path_pos[1] - current_pos[1])
            
            if dist >= lookahead:
                # Interpolate between current and next point
                if i > 0:
                    prev = self.path[i-1]
                    curr = self.path[i]
                    alpha = np.clip((lookahead - dist) / (dist + 1e-6), 0.0, 1.0)
                    
                    return State(
                        prev.theta_0 + alpha * (curr.theta_0 - prev.theta_0),
                        prev.theta_1 + alpha * (curr.theta_1 - prev.theta_1),
                        prev.omega_0 + alpha * (curr.omega_0 - prev.omega_0),
                        prev.omega_1 + alpha * (curr.omega_1 - prev.omega_1)
                    )
                return self.path[i]
        
        # If no point found at lookahead distance, use last valid point
        return self.path[min(self.path_index + look_ahead_window - 1, len(self.path) - 1)]

    def _update_path_index(self, robot: Robot) -> None:
        """Update path index by finding closest configuration in joint space"""
        if not self.path or self.path_index >= len(self.path):
            return
                
        try:
            # Get current joint angles
            current_theta_0 = robot.theta_0
            current_theta_1 = robot.theta_1

            # Debug: Print current tracking state
            self.debug_helper.log_state(f"\nPath Tracking State:")
            self.debug_helper.log_state(f"Current Index: {self.path_index}/{len(self.path)}")
            self.debug_helper.log_state(f"Current Position: θ0={current_theta_0:.3f}, θ1={current_theta_1:.3f}")
        
            
            min_dist = float('inf')
            closest_idx = self.path_index
            
            # Look ahead window with joint-space scaling
            look_ahead_window = min(
                10,  # Base window size
                len(self.path) - self.path_index  # Remaining path points
            )
            
            # Find closest configuration in joint space within window
            for i in range(self.path_index, min(self.path_index + look_ahead_window, len(self.path))):
                # Calculate joint-space distance
                delta_theta_0 = self._normalize_angle(self.path[i].theta_0 - current_theta_0)
                delta_theta_1 = self._normalize_angle(self.path[i].theta_1 - current_theta_1)
                
                # Use weighted joint-space metric considering joint ranges
                joint_0_weight = 1.0  # Can be adjusted based on joint importance
                joint_1_weight = 1.0
                
                # Compute joint-space distance with weights
                dist = np.sqrt(
                    joint_0_weight * delta_theta_0**2 + 
                    joint_1_weight * delta_theta_1**2
                )
                                   
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Only update if we're making forward progress and joint-space distance is reasonable
            if closest_idx > self.path_index: #and min_dist < np.pi/2:  # Half pi as max acceptable deviation
                self.path_index = closest_idx
            else:
                # If no better point found, stay at current point
                self.path_index = min(self.path_index + 1, len(self.path) - 1)
                
            # Debug: Print path update information
            if closest_idx > self.path_index:
                self.debug_helper.log_state(f"Updating path index: {self.path_index} -> {closest_idx}")
                self.debug_helper.log_state(f"Distance to reference: {min_dist:.3f} rad")
                self.path_index = closest_idx
            # Log large joint-space deviations
            if min_dist > 0.5:  # About 30 degrees total deviation
                self.debug_helper.log_state(f"Large joint-space tracking error: {min_dist:.3f} rad")
                
            self.mpc.set_path_index(self.path_index)
        except Exception as e:
            # Fallback: conservative increment with bounds check
            self.debug_helper.log_state(f"Error in path index update: {str(e)}")
            self.path_index = min(self.path_index, len(self.path) - 1)
            self.mpc.set_path_index(self.path_index)

    def _compute_lookahead_distance(self, robot: Robot) -> float:
        """Compute adaptive lookahead distance with improved scaling"""
        current_pos = robot.joint_2_pos()
        dist_to_goal = np.hypot(current_pos[0] - self.goal[0],
                               current_pos[1] - self.goal[1])
        
        # More conservative base lookahead
        base_lookahead = min(0.5, dist_to_goal)  # Reduced from 1.0
        
        # More conservative velocity scaling
        velocity = np.hypot(robot.omega_0, robot.omega_1)
        velocity_factor = 0.2 * velocity / self.constants.MAX_VELOCITY  # Reduced from 0.3
        
        # Increased error compensation
        if len(self.actual_theta_0) > 0:
            ref_idx = min(self.path_index, len(self.path)-1)
            tracking_error = np.hypot(
                self.path[ref_idx].theta_0 - robot.theta_0,
                self.path[ref_idx].theta_1 - robot.theta_1
            )
            error_factor = 0.5 * tracking_error  # Increased from 0.4
        else:
            error_factor = 0.0
                
        # Tighter bounds on lookahead distance
        lookahead = base_lookahead + velocity_factor - error_factor
        return np.clip(lookahead, 0.1, 1.0)  # Tighter bounds

    def _compute_current_state(self, robot: Robot) -> State:
        """Compute current state with improved reference validation"""
        if not self.path:
            return State(robot.theta_0, robot.theta_1, 0.0, 0.0)
            
        ref_state = self.compute_reference_state(robot)
        current_pos = robot.joint_2_pos()
        
        # Validate reference state
        if abs(ref_state.theta_0) > self.constants.JOINT_LIMITS[1] or \
           abs(ref_state.theta_1) > self.constants.JOINT_LIMITS[1]:
            self.debug_helper.log_state("Invalid reference state detected")
            # Return current state instead
            return State(robot.theta_0, robot.theta_1, robot.omega_0, robot.omega_1)
        
        return ref_state

    def _compute_tracking_errors(self, ref_state: State, robot: Robot) -> Dict[str, float]:
        """Compute tracking errors with validation"""
        errors = {
            'theta_0': self._normalize_angle(ref_state.theta_0 - robot.theta_0),
            'theta_1': self._normalize_angle(ref_state.theta_1 - robot.theta_1),
            'omega_0': ref_state.omega_0 - robot.omega_0,
            'omega_1': ref_state.omega_1 - robot.omega_1
        }
        
        # Check for excessive errors
        if abs(errors['theta_0']) > np.pi/2 or abs(errors['theta_1']) > np.pi/2:
            self.debug_helper.log_state(f"Warning: Large position error detected: {errors}")
            
        return errors

    def _compute_control_action(self, errors: Dict[str, float], robot: Robot) -> Dict[str, float]:
        """Compute control action with dynamic limit scaling based on error"""
        # Compute error magnitudes
        tracking_error = np.hypot(errors['theta_0'], errors['theta_1'])
        velocity_error = np.hypot(errors['omega_0'], errors['omega_1'])
        
        # Scale limits based on error magnitude
        error_scale = min(2.0, 1.0 + tracking_error)  # Allow up to 2x nominal limits
        max_accel = self.constants.MAX_ACCELERATION * error_scale
        max_vel = self.constants.MAX_VELOCITY * error_scale
        max_torque = 8000.0 * error_scale  # Increase torque limits proportionally
        
        # Compute adaptive gains
        kp = self._compute_adaptive_gain(self.base_kp, tracking_error) 
        kd = self._compute_adaptive_gain(self.base_kd, velocity_error)
        ki = self._compute_adaptive_gain(self.base_ki, tracking_error)
        
        # Update integral terms with improved anti-windup
        dt = self.constants.DT
        self.integral_error_0 = self._update_integral(self.integral_error_0, errors['theta_0'], dt)
        self.integral_error_1 = self._update_integral(self.integral_error_1, errors['theta_1'], dt)
        
        # Get reference states for feedforward
        if self.path_index < len(self.path):
            ref_state = self.path[self.path_index]
            next_index = min(self.path_index + 1, len(self.path) - 1)
            next_state = self.path[next_index]
            
            # Compute reference accelerations
            alpha_ref_0 = (next_state.omega_0 - ref_state.omega_0) / dt
            alpha_ref_1 = (next_state.omega_1 - ref_state.omega_1) / dt
            
            # Get dynamics matrices
            M, C, G = robot.compute_dynamics(
                robot.theta_0, robot.theta_1,
                robot.omega_0, robot.omega_1
            )
            
            # Compute feedforward torques with increased limits
            tau_ff_0, tau_ff_1 = robot.inverse_dynamics(
                ref_state.theta_0, ref_state.theta_1,
                ref_state.omega_0, ref_state.omega_1,
                alpha_ref_0,
                alpha_ref_1
            )
        else:
            tau_ff_0 = tau_ff_1 = 0.0
            alpha_ref_0 = alpha_ref_1 = 0.0
        
        # Compute feedback terms with error-based scaling
        scale = 0.1 * (max(1.0 , min(2.0, 1.0 + tracking_error)))  # Scale feedback with error
        u_fb_0 = scale * (kp * errors['theta_0'] + 
                         kd * errors['omega_0'] + 
                         ki * self.integral_error_0)
        u_fb_1 = scale * (kp * errors['theta_1'] + 
                         kd * errors['omega_1'] + 
                         ki * self.integral_error_1)
        
        # Combine feedback and feedforward
        ff_scale = self.feedforward_gain * (max(1.0 , min(2.0, 1.0 + tracking_error)))  # Scale feedforward compensation
        u_0 = u_fb_0 + ff_scale * tau_ff_0
        u_1 = u_fb_1 + ff_scale * tau_ff_1
        
        
        # Debug: Print control components
        self.debug_helper.log_state("\nControl Components:")
        self.debug_helper.log_state(f"Feedback: Joint 0: {u_fb_0:.3f}, Joint 1: {u_fb_1:.3f}")
        self.debug_helper.log_state(f"Feedforward: Joint 0: {tau_ff_0:.3f}, Joint 1: {tau_ff_1:.3f}")
        self.debug_helper.log_state(f"Total: Joint 0: {u_fb_0 + tau_ff_0:.3f}, Joint 1: {u_fb_1 + tau_ff_1:.3f}")
        # Add velocity damping with dynamic scaling
        damping = self._compute_adaptive_gain(5.0, tracking_error)
        u_0 -= damping * robot.omega_0
        u_1 -= damping * robot.omega_1
        
        # Apply scaled acceleration limits
        alpha_0 = np.clip(alpha_ref_0 + u_0, -max_accel, max_accel)
        alpha_1 = np.clip(alpha_ref_1 + u_1, -max_accel, max_accel)
        
        return {
            'alpha_0': alpha_0,
            'alpha_1': alpha_1,
            'feedback': {'joint0': u_fb_0, 'joint1': u_fb_1},
            'feedforward': {'joint0': tau_ff_0, 'joint1': tau_ff_1},
            'total': {'joint0': u_0, 'joint1': u_1}
        }

    def _apply_control(self, control: Dict[str, float], robot: Robot) -> Robot:
        """Apply control action with dynamic limit scaling"""
        dt = self.constants.DT
        
        # Calculate error-based scaling
        current_pos = robot.joint_2_pos()
        dist_to_goal = np.hypot(current_pos[0] - self.goal[0],
                               current_pos[1] - self.goal[1])
        error_scale = min(2.0, 1.0 + dist_to_goal)  # Allow up to 2x limits
        
        # Scale limits
        max_accel = self.constants.MAX_ACCELERATION * error_scale
        max_vel = self.constants.MAX_VELOCITY * error_scale
        max_torque = 5000.0 * error_scale
        
        # Get dynamics matrices
        M, C, G = robot.compute_dynamics(robot.theta_0, robot.theta_1, 
                                       robot.omega_0, robot.omega_1)
        
        # Apply scaled acceleration limits
        alpha_0 = np.clip(control['alpha_0'], -max_accel, max_accel)
        alpha_1 = np.clip(control['alpha_1'], -max_accel, max_accel)
        
        # Compute required torques
        tau = M @ np.array([alpha_0, alpha_1]) + C @ np.array([robot.omega_0, robot.omega_1]) + G
        
        # Apply scaled torque limits
        tau[0] = np.clip(tau[0], -max_torque, max_torque)
        tau[1] = np.clip(tau[1], -max_torque, max_torque)
        
        # Get resulting accelerations
        alpha_0, alpha_1 = robot.forward_dynamics(robot.theta_0, robot.theta_1,
                                                robot.omega_0, robot.omega_1,
                                                tau[0], tau[1])
        
        # Update velocities with scaled limiting
        omega_0_new = robot.omega_0 + alpha_0 * dt
        omega_1_new = robot.omega_1 + alpha_1 * dt
        
        # Apply scaled velocity limits with smooth saturation
        robot.omega_0 = self._smooth_velocity_limit(omega_0_new, max_vel)
        robot.omega_1 = self._smooth_velocity_limit(omega_1_new, max_vel)
        
        # Update positions with joint limit checking
        theta_0_new = robot.theta_0 + robot.omega_0 * dt
        theta_1_new = robot.theta_1 + robot.omega_1 * dt
        
        if self._check_joint_limits(theta_0_new, theta_1_new):
            robot.theta_0 = theta_0_new
            robot.theta_1 = theta_1_new
        else:
            # Reset velocities if joint limits are reached
            robot.omega_0 = 0.0
            robot.omega_1 = 0.0
        
        return robot

    def _handle_empty_path(self, robot: Robot) -> Robot:
        """Handle empty path case"""
        self.debug_helper.log_state("No path available or end of path reached")
        robot.omega_0 = 0.0
        robot.omega_1 = 0.0
        return robot

    def _handle_error(self, robot: Robot) -> Robot:
        """Handle controller error case"""
        self.debug_helper.log_state("Controller error - stopping robot")
        robot.omega_0 = 0.0
        robot.omega_1 = 0.0
        return robot

    def _update_tracking_history(self, robot: Robot) -> None:
        """Update tracking history with validation and acceleration computation"""
        try:
            # Store positions and velocities
            self.actual_theta_0.append(robot.theta_0)
            self.actual_theta_1.append(robot.theta_1)
            self.actual_omega_0.append(robot.omega_0)
            self.actual_omega_1.append(robot.omega_1)
            
            # Update reference velocities based on path
            if self.path and self.path_index < len(self.path):
                # Initialize reference lists if they don't exist
                if not hasattr(self, 'reference_omega_0') or not self.reference_omega_0:
                    self.reference_omega_0 = []
                    self.reference_omega_1 = []
                
                # Append current reference velocities
                self.reference_omega_0.append(self.path[self.path_index].omega_0)
                self.reference_omega_1.append(self.path[self.path_index].omega_1)
                
                # Keep reference and actual lists the same length
                if len(self.reference_omega_0) > len(self.actual_omega_0):
                    self.reference_omega_0 = self.reference_omega_0[:len(self.actual_omega_0)]
                    self.reference_omega_1 = self.reference_omega_1[:len(self.actual_omega_1)]
            
            # Compute accelerations with safeguards
            dt = self.constants.DT
            if len(self.actual_omega_0) >= 2:  # Need at least 2 points for acceleration
                alpha_0 = (self.actual_omega_0[-1] - self.actual_omega_0[-2]) / dt
                alpha_1 = (self.actual_omega_1[-1] - self.actual_omega_1[-2]) / dt
                
                # Bound accelerations to reasonable values
                max_accel = self.constants.MAX_ACCELERATION * 1.5  # Allow slight overshoot
                alpha_0 = np.clip(alpha_0, -max_accel, max_accel)
                alpha_1 = np.clip(alpha_1, -max_accel, max_accel)
                
                self.actual_alpha_0.append(alpha_0)
                self.actual_alpha_1.append(alpha_1)
            else:
                # Initialize with zero acceleration
                self.actual_alpha_0.append(0.0)
                self.actual_alpha_1.append(0.0)
            
            # Ensure all lists have the same length
            min_length = min(len(self.actual_theta_0), len(self.actual_omega_0), len(self.actual_alpha_0))
            self.actual_theta_0 = self.actual_theta_0[:min_length]
            self.actual_theta_1 = self.actual_theta_1[:min_length]
            self.actual_omega_0 = self.actual_omega_0[:min_length]
            self.actual_omega_1 = self.actual_omega_1[:min_length]
            self.actual_alpha_0 = self.actual_alpha_0[:min_length]
            self.actual_alpha_1 = self.actual_alpha_1[:min_length]
            
            # Update reference accelerations if needed
            if not self.reference_alpha_0:
                self.reference_alpha_0 = [0.0] * len(self.path)
                self.reference_alpha_1 = [0.0] * len(self.path)
                
                # Compute reference accelerations from velocity changes
                for i in range(1, len(self.path)-1):
                    self.reference_alpha_0[i] = (self.path[i+1].omega_0 - self.path[i-1].omega_0) / (2 * dt)
                    self.reference_alpha_1[i] = (self.path[i+1].omega_1 - self.path[i-1].omega_1) / (2 * dt)
                
                # Set first and last accelerations
                self.reference_alpha_0[0] = self.reference_alpha_0[1]
                self.reference_alpha_1[0] = self.reference_alpha_1[1]
                self.reference_alpha_0[-1] = self.reference_alpha_0[-2]
                self.reference_alpha_1[-1] = self.reference_alpha_1[-2]
            
            # Validate dynamics constraints periodically
            if len(self.actual_theta_0) % 50 == 0:
                self.debug_helper.validate_controller_dynamics(self, robot)
                
        except Exception as e:
            self.debug_helper.log_state(f"Error in tracking history update: {str(e)}")
    def _perform_debug_analysis(self, robot: Robot) -> None:
        """Perform periodic debugging analysis"""
        self.debug_helper.analyze_tracking_performance(self, robot)
        self.debug_helper.analyze_control_signals(self, robot)
        self.debug_helper.check_dynamic_consistency(self, robot)
        self.debug_helper.print_controller_stats(self, robot)
                
        # Update index if we found a closer point ahead
        if closest_idx > self.path_index:
            self.path_index = closest_idx


class VelocityProfileGenerator:
    def __init__(self, max_vel: float, max_acc: float, debug_helper=None):
        """Initialize velocity profile generator with constraints."""
        self.max_vel = abs(max_vel)
        self.max_acc = abs(max_acc)
        self.epsilon = 1e-6
        self.sub_profiles = []
        self.debug_helper = debug_helper
        
    def set_profile(self, init_pos: float, final_pos: float, init_vel: float, final_vel: float, init_time: float = 0.0) -> bool:
        """Generate velocity profile for given boundary conditions."""
        if self.debug_helper:
            self.debug_helper.log_state(f"\nVelocityProfile - Generating profile:")
            self.debug_helper.log_state(f"Initial position: {init_pos:.3f}")
            self.debug_helper.log_state(f"Final position: {final_pos:.3f}")
            self.debug_helper.log_state(f"Initial velocity: {init_vel:.3f}")
            self.debug_helper.log_state(f"Final velocity: {final_vel:.3f}")
            self.debug_helper.log_state(f"Max velocity: {self.max_vel:.3f}")
            self.debug_helper.log_state(f"Max acceleration: {self.max_acc:.3f}")

        # Reset profiles
        self.sub_profiles = []
        
        # Validate velocities
        if abs(init_vel) > self.max_vel or abs(final_vel) > self.max_vel:
            if self.debug_helper:
                self.debug_helper.log_state("ERROR: Initial/final velocities exceed maximum")
            return False
            
        # Calculate critical trajectory parameters
        delta_pos = final_pos - init_pos  # Distance to cover
        delta_vel = final_vel - init_vel  # Velocity change
        mean_vel = (final_vel + init_vel) / 2
        tf_crit = abs(delta_vel / self.max_acc)  # Critical time
        pos_crit = mean_vel * tf_crit  # Critical distance

        if self.debug_helper:
            self.debug_helper.log_state(f"\nCritical parameters:")
            self.debug_helper.log_state(f"Delta position: {delta_pos:.3f}")
            self.debug_helper.log_state(f"Delta velocity: {delta_vel:.3f}")
            self.debug_helper.log_state(f"Mean velocity: {mean_vel:.3f}")
            self.debug_helper.log_state(f"Critical time: {tf_crit:.3f}")
            self.debug_helper.log_state(f"Critical position: {pos_crit:.3f}")
        
        # Determine trajectory direction
        traj_sign = 0.0
        if delta_pos > pos_crit + self.epsilon:
            traj_sign = 1.0
        elif delta_pos < pos_crit - self.epsilon:
            traj_sign = -1.0
        elif delta_vel > self.epsilon:
            traj_sign = 1.0
        elif delta_vel < -self.epsilon:
            traj_sign = -1.0
            
        if self.debug_helper:
            self.debug_helper.log_state(f"\nTrajectory sign: {traj_sign}")
            
        if traj_sign == 0.0:
            if self.debug_helper:
                self.debug_helper.log_state("Target already reached - generating steady state profile")
            # Target already reached
            self._add_profile(init_time, init_pos, init_vel, 0.0)
            return True
            
        # Calculate peak velocity
        peak_vel = np.sqrt(abs(traj_sign * delta_pos * self.max_acc + 
                          (init_vel**2 + final_vel**2) / 2))
                          
        if self.debug_helper:
            self.debug_helper.log_state(f"Peak velocity: {peak_vel:.3f}")
                          
        if peak_vel < self.max_vel:
            if self.debug_helper:
                self.debug_helper.log_state("\nGenerating triangular profile")
            # Triangular profile
            T20 = (traj_sign * peak_vel - init_vel) / (traj_sign * self.max_acc)
            
            # Acceleration phase
            self._add_profile(init_time, init_pos, init_vel, traj_sign * self.max_acc)
            
            # Calculate intermediate position
            pos_T20 = init_pos + T20 * (init_vel + 0.5 * T20 * traj_sign * self.max_acc)
            
            if self.debug_helper:
                self.debug_helper.log_state(f"T20: {T20:.3f}")
                self.debug_helper.log_state(f"Intermediate position: {pos_T20:.3f}")
            
            # Deceleration phase  
            self._add_profile(init_time + T20, pos_T20, traj_sign * peak_vel, -traj_sign * self.max_acc)
            
        else:
            if self.debug_helper:
                self.debug_helper.log_state("\nGenerating trapezoidal profile")
            # Trapezoidal profile
            T20 = (traj_sign * self.max_vel - init_vel) / (traj_sign * self.max_acc)
            T30 = (1.0 / self.max_vel) * (traj_sign * delta_pos + 
                  (final_vel**2 + init_vel**2 - traj_sign * 2 * self.max_vel * init_vel) / (2 * self.max_acc))
            
            if self.debug_helper:
                self.debug_helper.log_state(f"T20: {T20:.3f}")
                self.debug_helper.log_state(f"T30: {T30:.3f}")
            
            # Acceleration phase
            self._add_profile(init_time, init_pos, init_vel, traj_sign * self.max_acc)
            
            # Calculate position at T20
            pos_T20 = init_pos + T20 * (init_vel + 0.5 * T20 * traj_sign * self.max_acc)
            
            # Constant velocity phase
            self._add_profile(init_time + T20, pos_T20, traj_sign * self.max_vel, 0.0)
            
            # Calculate position at T30
            pos_T30 = pos_T20 + (T30 - T20) * traj_sign * self.max_vel
            
            if self.debug_helper:
                self.debug_helper.log_state(f"Position at T20: {pos_T20:.3f}")
                self.debug_helper.log_state(f"Position at T30: {pos_T30:.3f}")
            
            # Deceleration phase
            self._add_profile(init_time + T30, pos_T30, traj_sign * self.max_vel, -traj_sign * self.max_acc)
            
        # Add final steady state profile
        final_time = self.duration + abs(final_vel) / self.max_acc
        final_position = final_pos + 0.5 * final_vel * abs(final_vel) / self.max_acc
        self._add_profile(final_time, final_position, 0.0, 0.0)

        if self.debug_helper:
            self.debug_helper.log_state(f"\nFinal profile parameters:")
            self.debug_helper.log_state(f"Final time: {final_time:.3f}")
            self.debug_helper.log_state(f"Final position: {final_position:.3f}")
            self.debug_helper.log_state(f"Number of sub-profiles: {len(self.sub_profiles)}")
            
        return True
        
    def _add_profile(self, t0: float, p0: float, v0: float, a: float):
        """Add a motion profile segment."""
        self.sub_profiles.append([t0, p0, v0, a])
        if self.debug_helper:
            self.debug_helper.log_state(f"Added profile segment: t0={t0:.3f}, p0={p0:.3f}, v0={v0:.3f}, a={a:.3f}")
        
    @property 
    def duration(self) -> float:
        """Get profile duration."""
        if not self.sub_profiles:
            return 0.0
        return self.sub_profiles[-1][0]
        
    def get_state_at_time(self, t: float) -> tuple:
        """Get position, velocity and acceleration at time t."""
        for i in range(len(self.sub_profiles) - 1):
            if t >= self.sub_profiles[i][0] and t < self.sub_profiles[i+1][0]:
                # Get profile parameters
                t0, p0, v0, a = self.sub_profiles[i]
                dt = t - t0
                
                # Calculate state
                p = p0 + v0*dt + 0.5*a*dt*dt
                v = v0 + a*dt
                return p, v, a
                
        # Return final state if beyond duration
        return (self.sub_profiles[-1][1], 
                self.sub_profiles[-1][2],
                self.sub_profiles[-1][3])

class PathParameterizer:
    """Converts discrete path points to continuous path with derivatives"""
    def __init__(self, path_points: List[State]):
        if not path_points:
            raise ValueError("Path points list cannot be empty")
        if len(path_points) < 2:
            raise ValueError("Need at least 2 path points for interpolation")
            
        self.original_points = path_points
        
        # Create more points for better interpolation
        num_points = max(len(path_points) * 2, 200)  # Ensure enough points for smooth interpolation
        self.s_points = np.linspace(0, 1, num_points)
        
        # Extract theta values and velocities
        self.theta_0_points = [p.theta_0 for p in path_points]
        self.theta_1_points = [p.theta_1 for p in path_points]
        self.omega_0_points = [p.omega_0 for p in path_points]
        self.omega_1_points = [p.omega_1 for p in path_points]
        
        # Original s values for the path points
        original_s = np.linspace(0, 1, len(path_points))
        
        # Create interpolators with proper boundary handling
        self.theta_0_spline = scipy.interpolate.CubicSpline(
            original_s, self.theta_0_points, 
            bc_type='clamped'
        )
        self.theta_1_spline = scipy.interpolate.CubicSpline(
            original_s, self.theta_1_points,
            bc_type='clamped'
        )
        
        # Create velocity interpolators
        self.omega_0_spline = scipy.interpolate.CubicSpline(
            original_s, self.omega_0_points,
            bc_type='clamped'
        )
        self.omega_1_spline = scipy.interpolate.CubicSpline(
            original_s, self.omega_1_points,
            bc_type='clamped'
        )

    def get_path_point(self, s: float) -> PathPoint:
        """Get path point and its derivatives at given s"""
        s = np.clip(s, 0, 1)  # Ensure s is in valid range
            
        # Get position
        theta_0 = float(self.theta_0_spline(s))
        theta_1 = float(self.theta_1_spline(s))
        
        # Get first derivatives (dtheta/ds)
        dtheta_0 = float(self.theta_0_spline.derivative(1)(s))
        dtheta_1 = float(self.theta_1_spline.derivative(1)(s))
        
        # Get second derivatives (d²theta/ds²)
        ddtheta_0 = float(self.theta_0_spline.derivative(2)(s))
        ddtheta_1 = float(self.theta_1_spline.derivative(2)(s))
        
        return PathPoint(s, theta_0, theta_1, dtheta_0, dtheta_1, ddtheta_0, ddtheta_1)

class TimeOptimalTrajectoryGenerator:
    def __init__(self, robot: Robot, path_points: List[State]):
        self.robot = robot
        self.tolerance = 1e-4  # Tolerance for binary search
        self.debug = DebugHelper(debug=True)
        self.debug.print_trajectory_header()
        # Initialize path parameterizer with the path points
        self.path = PathParameterizer(path_points)
        # Cache for dynamics calculations
        self.cached_dynamics = {}
        
    def compute_dynamics_at_s(self, s: float, omega_0: float = 0.0, omega_1: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute M, C, G matrices with caching"""
        
        if len(self.cached_dynamics) > 1000000:
            self._prune_cache()
        
        # Round to reduce cache misses from floating point errors
        cache_key = (round(s, 4), round(omega_0, 4), round(omega_1, 4))
        if cache_key in self.cached_dynamics:
            cached_M, cached_C, cached_G = self.cached_dynamics[cache_key]
            # self.debug.log_state(f"Using cached dynamics at s={s:.3f}:")
            # self.debug.log_state(f"  M matrix:\n{cached_M}")
            # self.debug.log_state(f"  C matrix:\n{cached_C}")
            # self.debug.log_state(f"  G vector:\n{cached_G}")
            return cached_M, cached_C, cached_G
            
        point = self.path.get_path_point(s)
        M, C, G = self.robot.compute_dynamics(point.theta_0, point.theta_1, omega_0, omega_1)
        
        # self.debug.log_state(f"Computing dynamics at s={s:.3f}:")
        # self.debug.log_state(f"  M matrix:\n{M}")
        # self.debug.log_state(f"  C matrix:\n{C}")
        # self.debug.log_state(f"  G vector:\n{G}")
        
        # Cache result
        self.cached_dynamics[cache_key] = (M, C, G)

       
        return M, C, G
        
    def _prune_cache(self):
        """
        Prunes the dynamics cache to prevent unbounded memory growth.
        Uses a combination of recency and relevance to determine which entries to keep.
        """
        # Maximum cache size
        MAX_CACHE_SIZE = 1000000
        REDUCTION_FACTOR = 0.75  # Remove 25% of cache when pruning
        TARGET_SIZE = int(MAX_CACHE_SIZE * REDUCTION_FACTOR)

        if len(self.cached_dynamics) <= TARGET_SIZE:
            return

        # Sort cache entries by their relevance score
        sorted_entries = sorted(
            self.cached_dynamics.items(),
            key=lambda x: self._compute_cache_score(x[0]),
            reverse=True  # Higher scores are more relevant
        )

        # Keep only the most relevant entries
        self.cached_dynamics = dict(sorted_entries[:TARGET_SIZE])

    def _compute_cache_score(self, cache_key: Tuple[float, float, float]) -> float:
        """
        Compute a relevance score for a cache entry.
        Higher scores mean the entry is more important to keep.
        
        Args:
            cache_key: Tuple of (s, omega_0, omega_1)
        
        Returns:
            float: Relevance score, higher is more relevant
        """
        s, omega_0, omega_1 = cache_key
        
        # Parameters for scoring
        s_weight = 1.0          # Weight for path parameter distance from endpoints
        velocity_weight = 0.5   # Weight for velocity magnitudes
        
        # Compute distance from endpoints (s=0 or s=1)
        # Points near endpoints are more important
        endpoint_distance = min(abs(s), abs(1.0 - s))
        s_score = 1.0 - endpoint_distance
        
        # Velocity score - lower velocities are more important 
        # as they're often critical points
        velocity_magnitude = np.sqrt(omega_0**2 + omega_1**2)
        velocity_score = 1.0 / (1.0 + velocity_magnitude)
        
        # Combine scores
        total_score = (s_weight * s_score + 
                      velocity_weight * velocity_score) / (s_weight + velocity_weight)
                      
        return total_score

    def compute_m(self, s: float) -> np.ndarray:
        """Compute m(s) from equation (9.32)"""
        point = self.path.get_path_point(s)
        M, _, _ = self.compute_dynamics_at_s(s)
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        m = M @ dtheta
        # self.debug.log_state(f"Computing m(s) at s={s:.3f}:")
        # self.debug.log_state(f"  dtheta: {dtheta}")
        # self.debug.log_state(f"  m(s): {m}")
        return m
    
    def compute_g(self, s: float) -> np.ndarray:
        """Get g(s) from dynamics"""
        _, _, G = self.compute_dynamics_at_s(s)
        return G

    def compute_christoffel_symbols(self, theta_0: float, theta_1: float) -> np.ndarray:
        """
        Compute Christoffel symbols from mass matrix partial derivatives.
        Returns a 2x2x2 tensor of Christoffel symbols.
        """
        # Get mass matrix at current configuration
        M, _, _ = self.robot.compute_dynamics(theta_0, theta_1, 0, 0)
        
        # Compute partial derivatives of mass matrix
        h = 1e-6  # Step size for numerical differentiation
        
        # ∂M/∂θ₀
        M_plus_0 = self.robot.compute_dynamics(theta_0 + h, theta_1, 0, 0)[0]
        M_minus_0 = self.robot.compute_dynamics(theta_0 - h, theta_1, 0, 0)[0]
        dM_dtheta0 = (M_plus_0 - M_minus_0) / (2*h)
        
        # ∂M/∂θ₁
        M_plus_1 = self.robot.compute_dynamics(theta_0, theta_1 + h, 0, 0)[0]
        M_minus_1 = self.robot.compute_dynamics(theta_0, theta_1 - h, 0, 0)[0]
        dM_dtheta1 = (M_plus_1 - M_minus_1) / (2*h)
        
        # Initialize Christoffel symbols tensor (2x2x2)
        Gamma = np.zeros((2, 2, 2))
        
        # Compute inverse mass matrix
        M_inv = np.linalg.inv(M)
        
        # Construct Christoffel symbols using the standard formula:
        # Γᵏᵢⱼ = 1/2 * M⁻¹ᵏˡ * (∂Mᵢₗ/∂qⱼ + ∂Mⱼₗ/∂qᵢ - ∂Mᵢⱼ/∂qₗ)
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        # Get partial derivatives
                        if l == 0:
                            dM_dq = dM_dtheta0
                        else:
                            dM_dq = dM_dtheta1
                            
                        term1 = dM_dq[i, l]  # ∂Mᵢₗ/∂qⱼ
                        term2 = dM_dq[j, l]  # ∂Mⱼₗ/∂qᵢ
                        
                        if j == 0:
                            term3 = dM_dtheta0[i, j]  # ∂Mᵢⱼ/∂qₗ
                        else:
                            term3 = dM_dtheta1[i, j]
                            
                        Gamma[k, i, j] += 0.5 * M_inv[k, l] * (term1 + term2 - term3)
        
        return Gamma

    def compute_c(self, s: float, s_dot: float) -> np.ndarray:
        # ------------ christoffel formula --------
        point = self.path.get_path_point(s)
        M, _, _ = self.compute_dynamics_at_s(s)
        
        # Get path derivatives
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        ddtheta = np.array([point.ddtheta_0, point.ddtheta_1])
        
        # Get Christoffel symbols at current configuration
        Gamma = self.compute_christoffel_symbols(point.theta_0, point.theta_1)
        
        # Compute M(θ)θ̈
        term1 = M @ ddtheta
        
        # Compute quadratic velocity terms using Christoffel symbols
        term2 = np.zeros(2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    term2[i] += Gamma[i, j, k] * dtheta[j] * dtheta[k]

        return term1 + term2
        # """Compute c(s) using relaxation formula"""
        # point = self.path.get_path_point(s)
        # dtheta = np.array([point.dtheta_0, point.dtheta_1])
        # ddtheta = np.array([point.ddtheta_0, point.ddtheta_1])
        # omega_0 = point.dtheta_0 * s_dot
        # omega_1 = point.dtheta_1 * s_dot
        
        # M, C, _ = self.compute_dynamics_at_s(s, omega_0, omega_1)
        
        # # Compute c(s) = M * θ''(s) + C * θ'(s) * s_dot
        # c = M @ ddtheta + C @ dtheta
        
        # # self.debug.log_state(f"Computing c(s) at s={s:.3f}, s_dot={s_dot:.3f}:")
        # # self.debug.log_state(f"  dtheta: {dtheta}")
        # # self.debug.log_state(f"  ddtheta: {ddtheta}")
        # # self.debug.log_state(f"  c(s): {c}")
        # return c
    
    def compute_limits(self, s: float, s_dot: float) -> Tuple[float, float]:
        """Compute L(s,s_dot) and U(s,s_dot) from equation (9.36)"""
        # self.debug.log_state(f"\nComputing limits at s={s:.3f}, s_dot={s_dot:.3f}")
        
        m = self.compute_m(s)
        c = self.compute_c(s, s_dot)
        g = self.compute_g(s)
        
        # self.debug.log_state(f"Components for limit calculation:")
        # self.debug.log_state(f"  m: {m}")
        # self.debug.log_state(f"  c: {c}")
        # self.debug.log_state(f"  g: {g}")
        
        # Initialize with the robot's acceleration limits
        L = -self.robot.constants.MAX_ACCELERATION
        U = self.robot.constants.MAX_ACCELERATION
        
        for i in range(len(m)):
            if abs(m[i]) < self.tolerance:
                # self.debug.log_state(f"  Joint {i}: Zero inertia case, skipping")
                continue
                
            c_term = c[i] * s_dot**2
            
            # Get velocity limits for current joint
            accel_max = self.robot.constants.MAX_ACCELERATION
            accel_min = -self.robot.constants.MAX_ACCELERATION
            
            if m[i] > 0:
                Li = (accel_min - c_term - g[i]) / m[i]
                Ui = (accel_max - c_term - g[i]) / m[i]
            else:
                Li = (accel_max - c_term - g[i]) / m[i]
                Ui = (accel_min - c_term - g[i]) / m[i]
                
            # self.debug.log_state(f"  Joint {i} limits:")
            # self.debug.log_state(f"    c_term: {c_term:.3f}")
            # self.debug.log_state(f"    Li: {Li:.3f}")
            # self.debug.log_state(f"    Ui: {Ui:.3f}")
                
            L = max(L, Li)
            U = min(U, Ui)
            
        # Ensure we don't exceed robot's acceleration limits
        L = max(L, -self.robot.constants.MAX_ACCELERATION)
        U = min(U, self.robot.constants.MAX_ACCELERATION)
        
        # self.debug.log_state(f"Final limits: L={L:.3f}, U={U:.3f}")
        return L, U
    
    def find_switch_points(self) -> List[Tuple[float, float, TrajectoryType]]:
        """Optimized switch point search with adaptive step size"""
        self.debug.log_state("\n=== Starting Switch Point Search ===")
        switches = []
        s_current = 0.0
        s_dot_current = 0.0
        
        # Start with larger initial step
        step_size = 0.15  
        min_step = 0.07
        growth_factor = 1.5
        reduction_factor = 0.5
        self.debug.log_state(f"Starting from (s={s_current}, s_dot={s_dot_current})")
        
        # Integrate backward curve once at start
        self.debug.log_state("\nStep 2: Integrating backward from end point")
        t_backward, backward = self.integrate_trajectory(1.0, 0.0, False, (0, 10.0))
        
        if backward.size == 0:
            self.debug.log_state("Warning: Backward integration produced no points")
            return []
            
        final_curve = backward
        self.debug.log_state(f"Generated {len(final_curve[0])} points on final curve")
        
        # Use adaptive step size
        iteration = 0
        max_iterations = 50  # Reduced max iterations
        
        while s_current < 1.0 and iteration < max_iterations:
            iteration += 1
            self.debug.log_state(f"\n=== Iteration {iteration} ===")
            self.debug.log_state(f"Current state: (s={s_current:.3f}, s_dot={s_dot_current:.3f})")
            self.debug.log_state(f"Current step size: {step_size:.3f}")
            
            # Forward integration
            self.debug.log_state("\nForward integration with maximum acceleration")
            t_forward, forward = self.integrate_trajectory(s_current, s_dot_current, True, (0, 10.0))
            
            # Check for intersection and violations
            intersection = self.detect_curve_intersection(forward, final_curve)
            violation = self.check_velocity_limit_violation(forward)
            
            if violation is None and intersection is not None:
                s_switch, s_dot_switch = intersection
                self.debug.log_state(f"Found intersection at (s={s_switch:.3f}, s_dot={s_dot_switch:.3f})")
                
                if abs(s_switch - s_current) > self.tolerance:
                    switches.append((s_switch, s_dot_switch, TrajectoryType.DECELERATION))
                    self.debug.log_state("Added deceleration switch point")
                    s_current = s_switch
                    s_dot_current = s_dot_switch
                    # No violation - increase step size
                    step_size = min(0.15, step_size * growth_factor)
                else:
                    self.debug.log_state("Switch point too close to current point, advancing")
                    s_current += step_size
                    s_dot_current = 0.0
                    
            elif violation is not None:
                s_lim, s_dot_lim = violation
                # Violation found - reduce step size and refine search
                step_size = max(min_step, step_size * reduction_factor)
                self.debug.log_state(f"Found velocity limit violation at (s={s_lim:.3f}, s_dot={s_dot_lim:.3f})")
                
                if s_lim <= s_current + self.tolerance:
                    self.debug.log_state("Violation point is behind or too close, advancing")
                    s_current += step_size
                    s_dot_current = 0.0
                    continue
                
                # Binary search for tangent point
                self.debug.log_state("\nPerforming binary search for tangent point")
                s_tan, s_dot_tan = self.binary_search_velocity(s_lim, s_dot_lim)
                self.debug.log_state(f"Found tangent point at (s={s_tan:.3f}, s_dot={s_dot_tan:.3f})")
                
                if abs(s_tan - s_current) <= self.tolerance:
                    self.debug.log_state("Tangent point too close to current point, advancing")
                    s_current += step_size
                    s_dot_current = 0.0
                    continue
                
                # Backward integration from tangent point
                self.debug.log_state("\nIntegrating backward from tangent point")
                t_back, back = self.integrate_trajectory(s_tan, s_dot_tan, False, (0, 10.0))
                
                intersection = self.detect_curve_intersection(back, forward)
                if intersection is not None:
                    s_switch, s_dot_switch = intersection
                    self.debug.log_state(f"Found intersection with forward curve at (s={s_switch:.3f}, s_dot={s_dot_switch:.3f})")
                    
                    if abs(s_switch - s_current) > self.tolerance:
                        switches.append((s_switch, s_dot_switch, TrajectoryType.DECELERATION))
                        switches.append((s_tan, s_dot_tan, TrajectoryType.ACCELERATION))
                        self.debug.log_state("Added deceleration and acceleration switch points")
                        s_current = s_tan
                        s_dot_current = s_dot_tan
                        step_size = max(min_step, step_size * 0.5)
                    else:
                        self.debug.log_state("Switch point too close to current point, advancing")
                        s_current += step_size
                        s_dot_current = 0.0
                else:
                    self.debug.log_state("No intersection found with forward curve, advancing")
                    s_current += step_size
                    s_dot_current = 0.0
                    step_size = min(0.15, step_size * 1.5)  # Increase step size when no intersection
            else:
                self.debug.log_state("No violation or intersection found, ending search")
                break
                
        self.debug.log_state(f"\nSwitch point search complete. Found {len(switches)} switch points:")
        for i, (s, s_dot, switch_type) in enumerate(switches):
            self.debug.log_state(f"Switch point {i+1}: (s={s:.3f}, s_dot={s_dot:.3f}, type={switch_type})")
            
        return switches

    def binary_search_velocity(self, s_lim: float, s_dot_lim: float) -> Tuple[float, float]:
        """Optimized binary search with early convergence"""
        self.debug.log_state(f"\nStarting binary search for velocity at s={s_lim:.3f}, s_dot_lim={s_dot_lim:.3f}")
        
        s_dot_low = 0.0
        s_dot_high = s_dot_lim
        best_point = None
        
        max_iterations = 20  # Reduced iterations
        rel_tolerance = 1e-3  # Relative tolerance for early stopping
        last_s_dot_test = s_dot_high;
        for iteration in range(max_iterations):
            s_dot_test = (s_dot_high + s_dot_low) / 2
            self.debug.log_state(f"\nBinary search iteration {iteration + 1}:")
            self.debug.log_state(f"  Testing s_dot={s_dot_test:.3f}")
            
            _, forward = self.integrate_trajectory(s_lim, s_dot_test, False, (0, 10.0))
            violation_point = self.check_velocity_limit_violation(forward)
            
            if violation_point is not None:
                self.debug.log_state(f"  Trajectory violates limits at {violation_point}")
                s_dot_high = s_dot_test
            else:
                self.debug.log_state(f"  Trajectory respects limits")
                s_dot_low = s_dot_test
                best_point = (s_lim, s_dot_test)
                
            self.debug.log_state(f"  Updated bounds: [{s_dot_low:.3f}, {s_dot_high:.3f}]")
            
            # Early stopping if converged enough
            if abs(s_dot_high - s_dot_low) < rel_tolerance * s_dot_lim or abs(s_dot_test - last_s_dot_test) < 5 * rel_tolerance:
                self.debug.log_state("  Converged within relative tolerance")
                break
            last_s_dot_test = s_dot_test
                
        final_point = best_point if best_point is not None else (s_lim, s_dot_low)
        self.debug.log_state(f"Binary search complete. Found point: (s={final_point[0]:.3f}, s_dot={final_point[1]:.3f})")
        return final_point

    def integrate_trajectory(self, s0: float, s_dot0: float, use_max: bool, t_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized trajectory integration with debug output"""
        self.debug.log_state(f"\nIntegrating trajectory from s0={s0:.3f}, s_dot0={s_dot0:.3f}")
        self.debug.log_state(f"Using {'maximum' if use_max else 'minimum'} acceleration")
        
        def dynamics(t: float, state: List[float]) -> List[float]:
            s, s_dot = state
            s = np.clip(s, 0, 1)
                
            try:
                L, U = self.compute_limits(s, s_dot)
                s_ddot = U if use_max else L
                s_ddot = np.clip(s_ddot, 
                               -self.robot.constants.MAX_ACCELERATION, 
                               self.robot.constants.MAX_ACCELERATION)
                               
                # self.debug.log_state(f"  t={t:.3f}: s={s:.3f}, s_dot={s_dot:.3f}, s_ddot={s_ddot:.3f}")
                return [s_dot, s_ddot]
            except Exception as e:
                self.debug.log_state(f"Error in dynamics: {e}")
                return [0.0, 0.0]

        # Adaptive step size based on path curvature
        def get_timestep(s):
            point = self.path.get_path_point(s)
            curvature = abs(point.ddtheta_0) + abs(point.ddtheta_1)
            return min(0.15, max(0.05, 0.15 / (1 + curvature)))
        
        self.debug.log_state("Starting numerical integration...")
        # Reduce number of evaluation points but keep fine resolution for debug
        t_eval = np.linspace(t_span[0], t_span[1], 50)
        
        solution = scipy.integrate.solve_ivp(
            dynamics,
            t_span,
            [s0, s_dot0],
            method='RK45',
            max_step=get_timestep(s0), # Increased max step size
            rtol=1e-2,     # Relaxed tolerance
            atol=1e-2     # Relaxed tolerance
            # t_eval=t_eval
        )
        
        if not solution.success:
            self.debug.log_state(f"Integration failed: {solution.message}")
            return np.array([t_span[0], t_span[1]]), np.array([[s0, s0], [s_dot0, 0.0]])
        
        self.debug.log_state(f"Integration complete: {len(solution.t)} points generated")
        
        # Filter points
        mask = (solution.y[0] >= 0) & (solution.y[0] <= 1)
        if not np.any(mask):
            self.debug.log_state("No points within valid range")
            return np.array([0, 1]), np.array([[s0, 1.0], [s_dot0, 0.0]])
            
        t_filtered = solution.t[mask]
        y_filtered = solution.y[:, mask]
        
        self.debug.log_state(f"Filtered trajectory: {len(t_filtered)} points")
        return t_filtered, y_filtered

    def detect_curve_intersection(self, curve1: np.ndarray, curve2: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect intersection between two phase-plane curves."""
        self.debug.log_state("\nSearching for curve intersection")
        
        # Ensure we have valid data
        if curve1.size == 0 or curve2.size == 0:
            self.debug.log_state("Invalid curve data")
            return None
            
        # Extract time and state arrays
        if len(curve1.shape) == 2 and curve1.shape[0] == 2:
            s1, s_dot1 = curve1[0], curve1[1]
        else:
            self.debug.log_state("Invalid shape for curve1")
            return None
            
        if len(curve2.shape) == 2 and curve2.shape[0] == 2:
            s2, s_dot2 = curve2[0], curve2[1]
        else:
            self.debug.log_state("Invalid shape for curve2")
            return None
            
        # Sort by s values
        idx1 = np.argsort(s1)
        idx2 = np.argsort(s2)
        s1, s_dot1 = s1[idx1], s_dot1[idx1]
        s2, s_dot2 = s2[idx2], s_dot2[idx2]
        
        # Find overlapping s range
        s_min = max(np.min(s1), np.min(s2))
        s_max = min(np.max(s1), np.max(s2))
        
        self.debug.log_state(f"Overlapping s range: [{s_min:.3f}, {s_max:.3f}]")
        
        if s_min >= s_max:
            self.debug.log_state("No overlapping range found")
            return None
            
        try:
            # Interpolate both curves
            f1 = scipy.interpolate.interp1d(s1, s_dot1, bounds_error=False)
            f2 = scipy.interpolate.interp1d(s2, s_dot2, bounds_error=False)
            
            # Create a fine grid of points
            s_points = np.linspace(s_min, s_max, 100)
            v1 = f1(s_points)
            v2 = f2(s_points)
            
            # Find where curves cross
            mask = ~np.isnan(v1) & ~np.isnan(v2)
            if not np.any(mask):
                self.debug.log_state("No valid intersection points found")
                return None
                
            s_points = s_points[mask]
            v1 = v1[mask]
            v2 = v2[mask]
            
            diff = v1 - v2
            zero_crossings = np.where(np.diff(np.signbit(diff)))[0]
            
            if len(zero_crossings) == 0:
                self.debug.log_state("No zero crossings found")
                return None
                
            # Get the first intersection point
            idx = zero_crossings[0]
            s_intersect = s_points[idx]
            s_dot_intersect = f1(s_intersect)
            
            self.debug.log_state(f"Found intersection at (s={s_intersect:.3f}, s_dot={s_dot_intersect:.3f})")
            return (float(s_intersect), float(s_dot_intersect))
            
        except Exception as e:
            self.debug.log_state(f"Error in intersection detection: {e}")
            return None

    def check_velocity_limit_violation(self, trajectory: np.ndarray) -> Optional[Tuple[float, float]]:
        """Check if trajectory violates velocity limits."""
        self.debug.log_state("\nChecking for velocity limit violations")
        
        for i in range(len(trajectory[0])):
            s = trajectory[0, i]
            s_dot = trajectory[1, i]
            L, U = self.compute_limits(s, s_dot)
            
            # Check if acceleration limits are violated
            if L > U or s_dot < 0:
                self.debug.log_state(f"Violation found at (s={s:.3f}, s_dot={s_dot:.3f})")
                self.debug.log_state(f"  Limits at violation: L={L:.3f}, U={U:.3f}")
                return (s, s_dot)
        
        self.debug.log_state("No velocity limit violations found")
        return None

    def compute_optimal_velocity_at_s(self, s: float, switch_points: List[Tuple[float, float, TrajectoryType]]) -> float:
        """Compute the optimal velocity at a given s based on switch points."""
        self.debug.log_state(f"\nComputing optimal velocity at s={s:.3f}")
        
        # Find the relevant phase between switch points
        current_phase = TrajectoryType.ACCELERATION
        s_prev = 0.0
        s_dot_prev = 0.0
        
        for s_switch, s_dot_switch, switch_type in switch_points:
            if s < s_switch:
                break
            s_prev = s_switch
            s_dot_prev = s_dot_switch
            current_phase = switch_type
            
        self.debug.log_state(f"Current phase: {current_phase}")
        self.debug.log_state(f"Previous switch point: (s={s_prev:.3f}, s_dot={s_dot_prev:.3f})")
        
        # Integrate from previous switch point to current s
        t_span = (0, 10.0)
        _, trajectory = self.integrate_trajectory(s_prev, s_dot_prev, 
                                               current_phase == TrajectoryType.ACCELERATION,
                                               t_span)
        
        # Find s_dot at current s by interpolation
        f = scipy.interpolate.interp1d(trajectory[0], trajectory[1])
        s_dot = float(f(s))
        
        self.debug.log_state(f"Computed optimal velocity: s_dot={s_dot:.3f}")
        return s_dot

    def generate_trajectory(self, path_points: List[State]) -> List[State]:
        """Generate time-optimal trajectory from path points."""        
        if not path_points:
            self.debug.log_state("Empty path points list")
            return []
            
        if len(path_points) == 1:
            self.debug.log_state("Single point trajectory, returning zero velocity")
            return [State(
                theta_0=path_points[0].theta_0,
                theta_1=path_points[0].theta_1,
                omega_0=0.0,
                omega_1=0.0
            )]
            
        try:
            # Preprocess path
            self.debug.log_state("\nPreprocessing path points...")
            cleaned_path = self.preprocess_path(path_points)
            self.debug.log_state(f"Path reduced from {len(path_points)} to {len(cleaned_path)} points")
            
            # Initialize path parameterizer
            self.debug.log_state("\nInitializing path parameterizer")
            self.path = PathParameterizer(cleaned_path)
            
            # Find switch points
            self.debug.log_state("\nFinding switch points...")
            switch_points = self.find_switch_points()
            self.debug.log_state(f"Found {len(switch_points)} switch points")
            
            # Generate final trajectory
            self.debug.log_state("\nGenerating final trajectory...")
            trajectory_points = []
            
            # Create more dense sampling near switch points
            s_values = []
            base_points = np.linspace(0, 1, max(100, len(cleaned_path) * 2))
            
            for s in base_points:
                s_values.append(s)
                # Add extra points near switch points
                for s_switch, _, _ in switch_points:
                    if abs(s - s_switch) < 0.05:
                        s_values.extend([
                            s_switch - 0.01,
                            s_switch,
                            s_switch + 0.01
                        ])
            
            s_values = sorted(list(set(np.clip(s_values, 0, 1))))
            self.debug.log_state(f"Generated {len(s_values)} trajectory points")
            
            for s in s_values:
                try:
                    point = self.path.get_path_point(s)
                    s_dot = self.compute_optimal_velocity_at_s(s, switch_points)
                    
                    # Convert path parameter velocity to joint velocities
                    omega_0 = point.dtheta_0 * s_dot
                    omega_1 = point.dtheta_1 * s_dot
                    
                    # Ensure velocities are within bounds
                    omega_0 = np.clip(omega_0, -self.robot.constants.MAX_VELOCITY, 
                                            self.robot.constants.MAX_VELOCITY)
                    omega_1 = np.clip(omega_1, -self.robot.constants.MAX_VELOCITY, 
                                            self.robot.constants.MAX_VELOCITY)
                    
                    trajectory_points.append(State(
                        theta_0=point.theta_0,
                        theta_1=point.theta_1,
                        omega_0=omega_0,
                        omega_1=omega_1
                    ))
                except Exception as e:
                    self.debug.log_state(f"Error generating trajectory point at s={s}: {e}")
                    continue
            
            if not trajectory_points:
                self.debug.log_state("Warning: Failed to generate any valid trajectory points")
                return cleaned_path
                
            self.debug.log_state(f"Successfully generated trajectory with {len(trajectory_points)} points")
            return trajectory_points
            
        except Exception as e:
            self.debug.log_state(f"Error in trajectory generation: {e}")
            return cleaned_path

    def preprocess_path(self, path_points: List[State]) -> List[State]:
        """
        Clean up path by removing duplicate or nearly identical points.
        Two points are considered nearly identical if their joint angles 
        differ by less than a small threshold.
        
        Args:
            path_points: Original list of path points
            
        Returns:
            List[State]: Cleaned path with duplicates removed
        """
        if not path_points:
            return []
            
        # Threshold for considering angles equal (in radians)
        angle_threshold = 0.01  
        
        # Start with first point
        cleaned_path = [path_points[0]]
        prev_point = path_points[0]
        
        for current_point in path_points[1:]:
            # Check if current point is significantly different from previous
            theta0_diff = abs(current_point.theta_0 - prev_point.theta_0)
            theta1_diff = abs(current_point.theta_1 - prev_point.theta_1)
            
            if theta0_diff > angle_threshold or theta1_diff > angle_threshold:
                cleaned_path.append(current_point)
                prev_point = current_point
                
        print(f"Path preprocessing: removed {len(path_points) - len(cleaned_path)} duplicate points")
        return cleaned_path

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
        self.final_velocities = (0.0, 0.0)  # Default final velocities

    def run(self) -> None:
        """
        Main simulation loop. Steps the controller and updates visualization.
        """
        running = True
        self.controller.set_goal(self.robot, self.goal, final_velocities=self.final_velocities)
        while running:
            self.robot = self.controller.step(self.robot)
            success = self.check_success(self.robot, self.goal)
            self.visualizer.update_display(self.robot, success, self.goal, controller=self.controller)
            # if success:
            #     running = False
            #     # Optionally, change final velocities for the next goal
            #     # For example, alternate between stopping and maintaining velocity
            #     # Here, we'll keep it simple and always set to zero
            #     self.final_velocities = (0.0, 0.0)
                
            #     # Generate a new random goal
            #     self.goal = generate_random_goal(
            #         self.constants.min_reachable_radius(),
            #         self.constants.max_reachable_radius()
            #     )
            #     self.controller.set_goal(self.robot, self.goal, final_velocities=self.final_velocities)
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
        position_threshold = 0.5
        velocity_threshold = 0.05
        
        current_pos = robot.joint_2_pos()
        position_error = np.hypot(current_pos[0] - goal[0], 
                                current_pos[1] - goal[1])
        
        position_close = position_error <= position_threshold
        velocity_close = (abs(robot.omega_0) <= velocity_threshold and 
                         abs(robot.omega_1) <= velocity_threshold)
        
        return position_close #and velocity_close

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

    # Add gravity consideration to config
    consider_gravity = True  # Set this to False for side-mounted links
    config['robot']['consider_gravity'] = consider_gravity

    # Initialize robot constants
    robot_constants = RobotConstants(config['robot'])

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
    #planner = SBPLLatticePlanner(robot, world)
    planner = AStarPlanner(robot, world)

    #

    # Initialize controller
    controller = Controller(robot, world, planner)

    # Initialize visualizer
    visualizer = Visualizer(world, config)

    runner = Runner(robot, controller, world, visualizer)

    try:
        runner.run()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Simulation aborted: {e}")
    finally:
        runner.cleanup()

# This DebugHelper class should be placed at the appropriate indentation level
class DebugHelper:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.planning_stats = {
            'iterations': 0,
            'explored_states': 0,
            'path_length': 0,
            'planning_time': 0.0,
            'start_time': None
        }
        self.tracking_history = []
        self.control_signals = []
        self.performance_metrics = {
            'position_rmse': [],
            'velocity_rmse': [],
            'control_effort': [],
            'tracking_delay': []
        }
        
    def print_planning_header(self):
        if self.debug:
            print("\n=== Starting Planning Phase ===")
            self.planning_stats['start_time'] = time.time()
            
    def print_trajectory_header(self):
        if self.debug:
            print("\n=== Starting Trajectory Generation ===")
            
    def print_controller_header(self):
        if self.debug:
            print("\n=== Starting Controller Execution ===")
            
    def log_state(self, msg: str):
        if self.debug:
            print(msg)
            
    def validate_path_limits(self, path: List[State], robot: Robot) -> bool:
        if not path:
            self.log_state("Empty path!")
            return False

        # Analyze joint angles
        for i, state in enumerate(path):
            # Check joint limits
            if not self.within_joint_limits(robot, (state.theta_0, state.theta_1)):
                self.log_state(f"Joint limit violation at index {i}:")
                self.log_state(f"θ0={state.theta_0:.3f}, θ1={state.theta_1:.3f}")
                return False

            # Check velocity limits
            if abs(state.omega_0) > robot.constants.MAX_VELOCITY or \
               abs(state.omega_1) > robot.constants.MAX_VELOCITY:
                self.log_state(f"Velocity limit violation at index {i}:")
                self.log_state(f"ω0={state.omega_0:.3f}, ω1={state.omega_1:.3f}")
                return False

            # Check accelerations if not first state
            if i > 0:
                alpha_0 = (state.omega_0 - path[i-1].omega_0) / robot.constants.DT
                alpha_1 = (state.omega_1 - path[i-1].omega_1) / robot.constants.DT
                
                if abs(alpha_0) > robot.constants.MAX_ACCELERATION or \
                   abs(alpha_1) > robot.constants.MAX_ACCELERATION:
                    self.log_state(f"Acceleration limit violation at index {i}:")
                    self.log_state(f"α0={alpha_0:.3f}, α1={alpha_1:.3f}")
                    return False
        return True

    
    def within_joint_limits(self, robot: Robot, node: Tuple[float, float]) -> bool:
        theta_0, theta_1 = node
        return (robot.constants.JOINT_LIMITS[0] <= theta_0 <= robot.constants.JOINT_LIMITS[1] and
                robot.constants.JOINT_LIMITS[0] <= theta_1 <= robot.constants.JOINT_LIMITS[1])
    def print_path_stats(self, path: List[State], robot: Robot):
        """Print detailed statistics about the planned path"""
        if not path:
            print("Empty path!")
            return
            
        print("\n=== A* Planner Output Statistics ===")
        print(f"Path length: {len(path)} points")
        
        # Angle ranges
        theta0_vals = [s.theta_0 for s in path]
        theta1_vals = [s.theta_1 for s in path]
        print(f"Theta 0 range: [{min(theta0_vals):.2f}, {max(theta0_vals):.2f}]")
        print(f"Theta 1 range: [{min(theta1_vals):.2f}, {max(theta1_vals):.2f}]")
        
        # Velocity ranges
        omega0_vals = [s.omega_0 for s in path]
        omega1_vals = [s.omega_1 for s in path]
        print(f"Omega 0 range: [{min(omega0_vals):.2f}, {max(omega0_vals):.2f}]")
        print(f"Omega 1 range: [{min(omega1_vals):.2f}, {max(omega1_vals):.2f}]")
        
        # Path smoothness
        theta0_diffs = np.diff(theta0_vals)
        theta1_diffs = np.diff(theta1_vals)
        print(f"Average angle change per step:")
        print(f"  Theta 0: {np.mean(np.abs(theta0_diffs)):.3f} rad")
        print(f"  Theta 1: {np.mean(np.abs(theta1_diffs)):.3f} rad")

    def validate_trajectory_dynamics(self, trajectory: List[State], robot: Robot) -> bool:
        """Validate if trajectory respects dynamic constraints"""
        if not trajectory:
            return False
            
        dt = robot.constants.DT
        max_accel = robot.constants.MAX_ACCELERATION
        
        for i in range(len(trajectory)-1):
            # Compute accelerations
            alpha_0 = (trajectory[i+1].omega_0 - trajectory[i].omega_0) / dt
            alpha_1 = (trajectory[i+1].omega_1 - trajectory[i].omega_1) / dt
            
            # Check acceleration limits
            if abs(alpha_0) > max_accel or abs(alpha_1) > max_accel:
                if self.debug:
                    print(f"Acceleration limits violated at step {i}:")
                    print(f"alpha0={alpha_0:.2f}, alpha1={alpha_1:.2f}")
                return False
                
            # Check dynamic feasibility
            tau_0, tau_1 = robot.inverse_dynamics(
                trajectory[i].theta_0, trajectory[i].theta_1,
                trajectory[i].omega_0, trajectory[i].omega_1,
                alpha_0, alpha_1
            )
            
            # Assuming torque limits of ±5.0
            if abs(tau_0) > 5.0 or abs(tau_1) > 5.0:
                if self.debug:
                    print(f"Torque limits violated at step {i}:")
                    print(f"tau0={tau_0:.2f}, tau1={tau_1:.2f}")
                return False
                
        return True

    def print_trajectory_stats(self, trajectory: List[State], robot: Robot):
        """Print detailed statistics about the time-optimal trajectory"""
        if not trajectory:
            print("Empty trajectory!")
            return
            
        print("\n=== Time-Optimal Trajectory Statistics ===")
        print(f"Trajectory length: {len(trajectory)} points")
        
        # Compute statistics for each joint
        dt = robot.constants.DT
        
        # Initialize arrays for stats
        theta0_vals = [s.theta_0 for s in trajectory]
        theta1_vals = [s.theta_1 for s in trajectory]
        omega0_vals = [s.omega_0 for s in trajectory]
        omega1_vals = [s.omega_1 for s in trajectory]
        alpha0_vals = np.diff(omega0_vals) / dt
        alpha1_vals = np.diff(omega1_vals) / dt
        
        # Print joint positions
        print("\nJoint Positions (rad):")
        print(f"Joint 0: min={min(theta0_vals):.3f}, max={max(theta0_vals):.3f}, "
              f"range={max(theta0_vals)-min(theta0_vals):.3f}")
        print(f"Joint 1: min={min(theta1_vals):.3f}, max={max(theta1_vals):.3f}, "
              f"range={max(theta1_vals)-min(theta1_vals):.3f}")
        
        # Print velocities
        print("\nJoint Velocities (rad/s):")
        print(f"Joint 0: min={min(omega0_vals):.3f}, max={max(omega0_vals):.3f}, "
              f"avg={np.mean(np.abs(omega0_vals)):.3f}, peak={max(np.abs(omega0_vals)):.3f}")
        print(f"Joint 1: min={min(omega1_vals):.3f}, max={max(omega1_vals):.3f}, "
              f"avg={np.mean(np.abs(omega1_vals)):.3f}, peak={max(np.abs(omega1_vals)):.3f}")
        
        # Print accelerations
        print("\nJoint Accelerations (rad/s²):")
        print(f"Joint 0: min={min(alpha0_vals):.3f}, max={max(alpha0_vals):.3f}, "
              f"avg={np.mean(np.abs(alpha0_vals)):.3f}, peak={max(np.abs(alpha0_vals)):.3f}")
        print(f"Joint 1: min={min(alpha1_vals):.3f}, max={max(alpha1_vals):.3f}, "
              f"avg={np.mean(np.abs(alpha1_vals)):.3f}, peak={max(np.abs(alpha1_vals)):.3f}")
        
        # Compute and print torques
        print("\nJoint Torques (Nm):")
        torques_0 = []
        torques_1 = []
        
        try:
            for i in range(len(trajectory)-1):
                alpha_0 = (trajectory[i+1].omega_0 - trajectory[i].omega_0) / dt
                alpha_1 = (trajectory[i+1].omega_1 - trajectory[i].omega_1) / dt
                tau_0, tau_1 = robot.inverse_dynamics(
                    trajectory[i].theta_0, trajectory[i].theta_1,
                    trajectory[i].omega_0, trajectory[i].omega_1,
                    alpha_0, alpha_1
                )
                torques_0.append(tau_0)
                torques_1.append(tau_1)
                
            print(f"Joint 0: min={min(torques_0):.3f}, max={max(torques_0):.3f}, "
                  f"avg={np.mean(np.abs(torques_0)):.3f}, peak={max(np.abs(torques_0)):.3f}")
            print(f"Joint 1: min={min(torques_1):.3f}, max={max(torques_1):.3f}, "
                  f"avg={np.mean(np.abs(torques_1)):.3f}, peak={max(np.abs(torques_1)):.3f}")
        except Exception as e:
            print(f"Error computing torques: {str(e)}")
            
        # Check for limit violations
        print("\nLimit Checks:")
        vel_limit = robot.constants.MAX_VELOCITY
        acc_limit = robot.constants.MAX_ACCELERATION
        torque_limit = 1000.0  # Assumed torque limit
        
        vel_violations = sum(1 for v in omega0_vals + omega1_vals if abs(v) > vel_limit)
        acc_violations = sum(1 for a in alpha0_vals + alpha1_vals if abs(a) > acc_limit)
        torque_violations = sum(1 for t in torques_0 + torques_1 if abs(t) > torque_limit)
        
        print(f"Velocity limit ({vel_limit:.2f} rad/s) violations: {vel_violations}")
        print(f"Acceleration limit ({acc_limit:.2f} rad/s²) violations: {acc_violations}")
        print(f"Torque limit ({torque_limit:.2f} Nm) violations: {torque_violations}")
        
        # Print trajectory points if number is reasonable
        if len(trajectory) <= 20:  # Only print if trajectory is short
            print("\nDetailed Trajectory Points:")
            print("  idx    theta0    theta1    omega0    omega1    alpha0    alpha1     tau0     tau1")
            print("  ---    ------    ------    ------    ------    ------    ------     ----     ----")
            for i in range(len(trajectory)-1):
                alpha_0 = alpha0_vals[i]
                alpha_1 = alpha1_vals[i]
                tau_0 = torques_0[i] if torques_0 else 0
                tau_1 = torques_1[i] if torques_1 else 0
                print(f"  {i:3d}  {trajectory[i].theta_0:8.3f}  {trajectory[i].theta_1:8.3f}  "
                      f"{trajectory[i].omega_0:8.3f}  {trajectory[i].omega_1:8.3f}  "
                      f"{alpha_0:8.3f}  {alpha_1:8.3f}  {tau_0:8.3f}  {tau_1:8.3f}")


    def validate_planner_output(self, path: List[State], start: Tuple[float, float], 
                              goal: Tuple[float, float], robot: Robot) -> bool:
        """Validate path output from planner"""
        if not path:
            if self.debug:
                print("ERROR: Empty path returned from planner")
            return False

        # Check start position
        start_pos = robot.forward_kinematics(path[0].theta_0, path[0].theta_1)
        start_error = np.hypot(start_pos[0] - start[0], start_pos[1] - start[1])

        if start_error > 0.1:
            if self.debug:
                print(f"ERROR: Path doesn't start at robot position. Error: {start_error:.3f}")
            return False

        # Check goal reaching
        end_pos = robot.forward_kinematics(path[-1].theta_0, path[-1].theta_1)
        goal_error = np.hypot(end_pos[0] - goal[0], end_pos[1] - goal[1])

        if goal_error > 0.5:
            # if self.debug:
            #     print(f"WARNING: Path ends far from goal. Error: {goal_error:.3f}")
            return False

        # Validate limits
        return self.validate_path_limits(path, robot)

    
    def analyze_tracking_performance(self, controller, robot):
        """Detailed analysis of tracking performance"""
        if not controller.actual_theta_0 or not controller.reference_theta_0:
            print("No tracking data available")
            return

        print("\n=== Tracking Performance Analysis ===")
        
        # Compute tracking metrics for each joint
        min_len = min(len(controller.actual_theta_0), len(controller.reference_theta_0))
        
        # Position tracking analysis
        theta0_error = np.array(controller.actual_theta_0[:min_len]) - np.array(controller.reference_theta_0[:min_len])
        theta1_error = np.array(controller.actual_theta_1[:min_len]) - np.array(controller.reference_theta_1[:min_len])
        
        # End-effector position error
        ee_actual = [robot.forward_kinematics(t0, t1) for t0, t1 in 
                    zip(controller.actual_theta_0[:min_len], controller.actual_theta_1[:min_len])]
        ee_ref = [robot.forward_kinematics(t0, t1) for t0, t1 in 
                 zip(controller.reference_theta_0[:min_len], controller.reference_theta_1[:min_len])]
        ee_error = [np.hypot(a[0]-r[0], a[1]-r[1]) for a, r in zip(ee_actual, ee_ref)]
        
        print("\nPosition Tracking Metrics:")
        print(f"Joint 0 RMS Error: {np.sqrt(np.mean(theta0_error**2)):.4f} rad")
        print(f"Joint 1 RMS Error: {np.sqrt(np.mean(theta1_error**2)):.4f} rad")
        print(f"End-effector RMS Error: {np.sqrt(np.mean(np.array(ee_error)**2)):.4f} units")
        
        # Velocity tracking analysis
        if controller.actual_omega_0 and controller.reference_omega_0:
            min_len = min(len(controller.actual_omega_0), len(controller.reference_omega_0))
            omega0_error = np.array(controller.actual_omega_0[:min_len]) - np.array(controller.reference_omega_0[:min_len])
            omega1_error = np.array(controller.actual_omega_1[:min_len]) - np.array(controller.reference_omega_1[:min_len])
            
            print("\nVelocity Tracking Metrics:")
            print(f"Joint 0 Velocity RMS Error: {np.sqrt(np.mean(omega0_error**2)):.4f} rad/s")
            print(f"Joint 1 Velocity RMS Error: {np.sqrt(np.mean(omega1_error**2)):.4f} rad/s")
        
        # Analyze tracking delay
        delay0 = self._estimate_tracking_delay(controller.reference_theta_0[:min_len], 
                                             controller.actual_theta_0[:min_len])
        delay1 = self._estimate_tracking_delay(controller.reference_theta_1[:min_len], 
                                             controller.actual_theta_1[:min_len])
        
        print("\nTracking Delay Analysis:")
        print(f"Estimated Joint 0 Delay: {delay0:.2f} timesteps")
        print(f"Estimated Joint 1 Delay: {delay1:.2f} timesteps")

    def _estimate_tracking_delay(self, reference, actual, max_delay=20):
        """Estimate tracking delay using cross-correlation"""
        if len(reference) < max_delay * 2:
            return 0
            
        correlations = []
        for delay in range(max_delay):
            correlation = np.corrcoef(reference[delay:], actual[:-delay if delay > 0 else None])[0,1]
            correlations.append(correlation)
        
        return np.argmax(correlations)

    def analyze_control_signals(self, controller, robot):
        """Analyze control signal characteristics"""
        print("\n=== Control Signal Analysis ===")
        
        # Compute control efforts (torques)
        dt = robot.constants.DT
        torques_0 = []
        torques_1 = []
        
        try:
            for i in range(len(controller.actual_theta_0)-1):
                if (i < len(controller.actual_omega_0) and i+1 < len(controller.actual_omega_0)):
                    # Compute accelerations
                    alpha_0 = (controller.actual_omega_0[i+1] - controller.actual_omega_0[i]) / dt
                    alpha_1 = (controller.actual_omega_1[i+1] - controller.actual_omega_1[i]) / dt
                    
                    # Get required torques
                    tau_0, tau_1 = robot.inverse_dynamics(
                        controller.actual_theta_0[i],
                        controller.actual_theta_1[i],
                        controller.actual_omega_0[i],
                        controller.actual_omega_1[i],
                        alpha_0, alpha_1
                    )
                    torques_0.append(tau_0)
                    torques_1.append(tau_1)
            
            if torques_0:
                # Analyze control signal characteristics
                print("\nTorque Statistics:")
                print(f"Joint 0 Mean Torque: {np.mean(torques_0):.3f} Nm")
                print(f"Joint 1 Mean Torque: {np.mean(torques_1):.3f} Nm")
                print(f"Joint 0 Torque Std: {np.std(torques_0):.3f} Nm")
                print(f"Joint 1 Torque Std: {np.std(torques_1):.3f} Nm")
                print(f"Joint 0 Peak Torque: {max(abs(np.min(torques_0)), abs(np.max(torques_0))):.3f} Nm")
                print(f"Joint 1 Peak Torque: {max(abs(np.min(torques_1)), abs(np.max(torques_1))):.3f} Nm")
                
                # Analyze control signal smoothness
                torque_derivatives_0 = np.diff(torques_0) / dt
                torque_derivatives_1 = np.diff(torques_1) / dt
                
                print("\nControl Signal Smoothness:")
                print(f"Joint 0 Mean Torque Rate: {np.mean(abs(torque_derivatives_0)):.3f} Nm/s")
                print(f"Joint 1 Mean Torque Rate: {np.mean(abs(torque_derivatives_1)):.3f} Nm/s")
                print(f"Joint 0 Peak Torque Rate: {max(abs(torque_derivatives_0)):.3f} Nm/s")
                print(f"Joint 1 Peak Torque Rate: {max(abs(torque_derivatives_1)):.3f} Nm/s")
                
                # Check for saturation
                torque_limit = 5.0  # Assumed torque limit
                saturation_0 = sum(1 for t in torques_0 if abs(t) >= torque_limit * 0.95)
                saturation_1 = sum(1 for t in torques_1 if abs(t) >= torque_limit * 0.95)
                
                print(f"\nTorque Saturation Analysis:")
                print(f"Joint 0 Saturation Events: {saturation_0} ({saturation_0/len(torques_0)*100:.1f}%)")
                print(f"Joint 1 Saturation Events: {saturation_1} ({saturation_1/len(torques_1)*100:.1f}%)")
        
        except Exception as e:
            print(f"Error in control signal analysis: {str(e)}")

    def analyze_path_tracking(self, controller, robot):
        """Analyze path tracking behavior"""
        print("\n=== Path Tracking Analysis ===")
        
        if not controller.path or not controller.actual_theta_0:
            print("No path tracking data available")
            return
            
        try:
            # Compute actual path in workspace
            actual_positions = [robot.forward_kinematics(t0, t1) 
                              for t0, t1 in zip(controller.actual_theta_0, controller.actual_theta_1)]
            
            # Compute reference path in workspace
            reference_positions = [robot.forward_kinematics(state.theta_0, state.theta_1) 
                                 for state in controller.path]
            
            # Analyze path deviation
            min_len = min(len(actual_positions), len(reference_positions))
            path_errors = [np.hypot(a[0]-r[0], a[1]-r[1]) 
                          for a, r in zip(actual_positions[:min_len], reference_positions[:min_len])]
            
            print("\nPath Deviation Metrics:")
            print(f"Mean Path Error: {np.mean(path_errors):.4f} units")
            print(f"Max Path Error: {np.max(path_errors):.4f} units")
            print(f"Path Error Std: {np.std(path_errors):.4f} units")
            
            # Analyze path smoothness
            if len(actual_positions) > 1:
                actual_velocities = np.diff(actual_positions, axis=0) / robot.constants.DT
                actual_speed = np.hypot(actual_velocities[:,0], actual_velocities[:,1])
                
                print("\nPath Smoothness Metrics:")
                print(f"Mean Speed: {np.mean(actual_speed):.4f} units/s")
                print(f"Speed Variation: {np.std(actual_speed):.4f} units/s")
                
                # Compute curvature
                if len(actual_positions) > 2:
                    dx = np.diff([p[0] for p in actual_positions])
                    dy = np.diff([p[1] for p in actual_positions])
                    ddx = np.diff(dx)
                    ddy = np.diff(dy)
                    curvature = np.abs(dx[:-1]*ddy - dy[:-1]*ddx) / (dx[:-1]**2 + dy[:-1]**2)**1.5
                    
                    print(f"Mean Path Curvature: {np.mean(curvature):.4f}")
                    print(f"Max Path Curvature: {np.max(curvature):.4f}")
            
            # Analyze tracking progress
            if hasattr(controller, 'path_index'):
                progress_rate = controller.path_index / len(controller.path)
                print(f"\nPath Progress: {progress_rate*100:.1f}% complete")
                
        except Exception as e:
            print(f"Error in path tracking analysis: {str(e)}")

    def check_dynamic_consistency(self, controller, robot):
        """Check if motion satisfies dynamic constraints"""
        print("\n=== Dynamic Consistency Check ===")
        
        try:
            dt = robot.constants.DT
            max_accel = robot.constants.MAX_ACCELERATION
            max_vel = robot.constants.MAX_VELOCITY
            violations = {
                'velocity': [],
                'acceleration': [],
                'torque': []
            }
            
            for i in range(len(controller.actual_theta_0)-1):
                # Check velocity limits
                if abs(controller.actual_omega_0[i]) > max_vel:
                    violations['velocity'].append(('Joint 0', i, controller.actual_omega_0[i]))
                if abs(controller.actual_omega_1[i]) > max_vel:
                    violations['velocity'].append(('Joint 1', i, controller.actual_omega_1[i]))
                
                # Check acceleration limits
                if i < len(controller.actual_omega_0)-1:
                    alpha_0 = (controller.actual_omega_0[i+1] - controller.actual_omega_0[i]) / dt
                    alpha_1 = (controller.actual_omega_1[i+1] - controller.actual_omega_1[i]) / dt
                    
                    if abs(alpha_0) > max_accel:
                        violations['acceleration'].append(('Joint 0', i, alpha_0))
                    if abs(alpha_1) > max_accel:
                        violations['acceleration'].append(('Joint 1', i, alpha_1))
                    
                    # Check torque limits
                    tau_0, tau_1 = robot.inverse_dynamics(
                        controller.actual_theta_0[i],
                        controller.actual_theta_1[i],
                        controller.actual_omega_0[i],
                        controller.actual_omega_1[i],
                        alpha_0, alpha_1
                    )
                    
                    torque_limit = 5.0
                    if abs(tau_0) > torque_limit:
                        violations['torque'].append(('Joint 0', i, tau_0))
                    if abs(tau_1) > torque_limit:
                        violations['torque'].append(('Joint 1', i, tau_1))
            
            # Report violations
            print("\nConstraint Violations:")
            for constraint_type, violation_list in violations.items():
                if violation_list:
                    print(f"\n{constraint_type.capitalize()} Violations:")
                    for joint, timestep, value in violation_list[:5]:  # Show first 5 violations
                        print(f"  {joint} at step {timestep}: {value:.3f}")
                    if len(violation_list) > 5:
                        print(f"  ... and {len(violation_list)-5} more violations")
                else:
                    print(f"\nNo {constraint_type} violations detected")
                    
        except Exception as e:
            print(f"Error in dynamic consistency check: {str(e)}")

    def validate_controller_dynamics(self, controller, robot: Robot) -> bool:
        """Validate if controller respects dynamic constraints"""
        if not controller.actual_theta_0:
            return True
            
        dt = robot.constants.DT
        max_accel = robot.constants.MAX_ACCELERATION
        
        for i in range(len(controller.actual_theta_0)-1):
            # Compute actual accelerations
            alpha_0 = (controller.actual_omega_0[i+1] - controller.actual_omega_0[i]) / dt
            alpha_1 = (controller.actual_omega_1[i+1] - controller.actual_omega_1[i]) / dt
            
            # Check acceleration limits with 10% tolerance
            if abs(alpha_0) > max_accel * 1.1 or abs(alpha_1) > max_accel * 1.1:
                if self.debug:
                    print(f"Controller acceleration limits violated at step {i}")
                return False
                
            # Validate joint limits
            if not (robot.constants.JOINT_LIMITS[0] <= controller.actual_theta_0[i] <= robot.constants.JOINT_LIMITS[1] and
                   robot.constants.JOINT_LIMITS[0] <= controller.actual_theta_1[i] <= robot.constants.JOINT_LIMITS[1]):
                if self.debug:
                    print(f"Controller joint limits violated at step {i}")
                return False
                
        return True

    def print_controller_stats(self, controller, robot: Robot):
        """Print detailed statistics about the controller performance"""
        print("\n=== Controller Performance Statistics ===")
        
        # Check if we have enough data points
        if (not hasattr(controller, 'actual_theta_0') or 
            not controller.actual_theta_0 or 
            len(controller.actual_theta_0) < 2):
            print("Insufficient controller data available!")
            return
            
        try:
            # Tracking errors - ensure arrays are the same length
            min_len = min(len(controller.actual_theta_0), len(controller.reference_theta_0))
            if min_len > 0:
                theta0_errors = np.array(controller.actual_theta_0[:min_len]) - np.array(controller.reference_theta_0[:min_len])
                theta1_errors = np.array(controller.actual_theta_1[:min_len]) - np.array(controller.reference_theta_1[:min_len])
                
                print(f"RMS Position Tracking Errors:")
                print(f"  Joint 0: {np.sqrt(np.mean(theta0_errors**2)):.3f} rad")
                print(f"  Joint 1: {np.sqrt(np.mean(theta1_errors**2)):.3f} rad")
            
                # Velocity tracking
                min_len = min(len(controller.actual_omega_0), len(controller.reference_omega_0))
                omega0_errors = np.array(controller.actual_omega_0[:min_len]) - np.array(controller.reference_omega_0[:min_len])
                omega1_errors = np.array(controller.actual_omega_1[:min_len]) - np.array(controller.reference_omega_1[:min_len])
                
                print(f"RMS Velocity Tracking Errors:")
                print(f"  Joint 0: {np.sqrt(np.mean(omega0_errors**2)):.3f} rad/s")
                print(f"  Joint 1: {np.sqrt(np.mean(omega1_errors**2)):.3f} rad/s")
            
            # Compute control efforts (torques)
            dt = robot.constants.DT
            torques_0 = []
            torques_1 = []
            
            for i in range(len(controller.actual_theta_0)-1):
                if (i < len(controller.actual_omega_0) and 
                    i+1 < len(controller.actual_omega_0)):
                    
                    alpha_0 = (controller.actual_omega_0[i+1] - controller.actual_omega_0[i]) / dt
                    alpha_1 = (controller.actual_omega_1[i+1] - controller.actual_omega_1[i]) / dt
                    
                    tau_0, tau_1 = robot.inverse_dynamics(
                        controller.actual_theta_0[i],
                        controller.actual_theta_1[i],
                        controller.actual_omega_0[i],
                        controller.actual_omega_1[i],
                        alpha_0, alpha_1
                    )
                    torques_0.append(tau_0)
                    torques_1.append(tau_1)
            
            print(f"\nControl Effort Statistics:")
            if torques_0:  # Only print if we have torque data
                print(f"Average absolute torques:")
                print(f"  Joint 0: {np.mean(np.abs(torques_0)):.2f} Nm")
                print(f"  Joint 1: {np.mean(np.abs(torques_1)):.2f} Nm")
                print(f"Peak torques:")
                print(f"  Joint 0: {max(abs(min(torques_0)), abs(max(torques_0))):.2f} Nm")
                print(f"  Joint 1: {max(abs(min(torques_1)), abs(max(torques_1))):.2f} Nm")
            else:
                print("No torque data available yet")
        except Exception as e:
            print(f"Error computing controller statistics: {str(e)}")

    def print_path_points(self, path: List[State]):
        """Print detailed path point information"""
        print("\n=== Path Points ===")
        print("idx, theta_0, theta_1, omega_0, omega_1")
        for i, state in enumerate(path):
            print(f"{i:3d}, {state.theta_0:7.3f}, {state.theta_1:7.3f}, {state.omega_0:7.3f}, {state.omega_1:7.3f}")

    def print_trajectory_points(self, trajectory: List[State], robot: Robot):
        """Print detailed trajectory point information"""
        print("\n=== Trajectory Points ===")
        print("idx, theta_0, theta_1, omega_0, omega_1, alpha_0, alpha_1")
        
        dt = robot.constants.DT
        for i in range(len(trajectory)-1):
            alpha_0 = (trajectory[i+1].omega_0 - trajectory[i].omega_0) / dt
            alpha_1 = (trajectory[i+1].omega_1 - trajectory[i].omega_1) / dt
            print(f"{i:3d}, {trajectory[i].theta_0:7.3f}, {trajectory[i].theta_1:7.3f}, "
                  f"{trajectory[i].omega_0:7.3f}, {trajectory[i].omega_1:7.3f}, "
                  f"{alpha_0:7.3f}, {alpha_1:7.3f}")

    def print_step_details(self, controller, robot, control_actions):
        """Print detailed information about the current control step"""
        print("\n=== Step Details ===")
        
        # Current state
        print(f"Current State:")
        print(f"  Joint Positions: θ0={robot.theta_0:.3f}, θ1={robot.theta_1:.3f}")
        print(f"  Joint Velocities: ω0={robot.omega_0:.3f}, ω1={robot.omega_1:.3f}")
        
        # Path tracking
        print(f"\nPath Tracking:")
        print(f"  Current Path Index: {controller.path_index}")
        print(f"  Total Path Length: {len(controller.path)}")
        
        # Reference state
        if controller.path and controller.path_index < len(controller.path):
            ref_state = controller.path[controller.path_index]
            print(f"\nReference State:")
            print(f"  Reference Positions: θ0={ref_state.theta_0:.3f}, θ1={ref_state.theta_1:.3f}")
            print(f"  Reference Velocities: ω0={ref_state.omega_0:.3f}, ω1={ref_state.omega_1:.3f}")
            
            # Position errors
            pos_error_0 = ref_state.theta_0 - robot.theta_0
            pos_error_1 = ref_state.theta_1 - robot.theta_1
            print(f"\nTracking Errors:")
            print(f"  Position Errors: e0={pos_error_0:.3f}, e1={pos_error_1:.3f}")
            print(f"  Velocity Errors: eω0={ref_state.omega_0 - robot.omega_0:.3f}, "
                  f"eω1={ref_state.omega_1 - robot.omega_1:.3f}")
        
        # Control actions
        if control_actions:
            print(f"\nControl Components:")
            for component, values in control_actions.items():
                if isinstance(values, dict):
                    print(f"  {component}:")
                    for key, value in values.items():
                        print(f"    {key}: {value:.3f}")
                else:
                    print(f"  {component}: {values:.3f}")
            
            # Compute control contribution percentages
            if 'total' in control_actions and control_actions['total'] != 0:
                print(f"\nControl Contribution Percentages:")
                for component in ['feedback', 'feedforward']:
                    if component in control_actions:
                        contribution = (control_actions[component] / control_actions['total']) * 100
                        print(f"  {component}: {contribution:.1f}%")
        
        print("\n" + "="*50)  # Separator for readability

    def print_planning_header(self):
        if self.debug:
            print("\n=== Starting Planning Phase ===")
            self.planning_stats['start_time'] = time.time()
            
    def print_trajectory_header(self):
        if self.debug:
            print("\n=== Starting Trajectory Generation ===")
            
    def print_controller_header(self):
        if self.debug:
            print("\n=== Starting Controller Execution ===")
            
    def log_state(self, msg: str):
        if self.debug:
            print(msg)

if __name__ == "__main__":
    main()
