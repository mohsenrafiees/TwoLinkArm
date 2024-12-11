from typing import Tuple, Optional, List
import pygame
import numpy as np
from enviroment.robot import Robot
from enviroment.world import World
from utils.debug_helper import DebugHelper
from utils.state import State
from controller.controller import Controller

class Visualizer:
    """
    A visualization module for rendering 2D robot simulation using pygame.
    
    This class handles all visualization aspects including:
    - Robot visualization (joints and links)
    - Environment rendering (obstacles and goals)
    - Path visualization (planned and executed)
    - Performance metrics plotting (velocities and accelerations)
    """

    def __init__(self, world: World, config: dict) -> None:
        """
        Initialize the visualization environment with pygame.

        Args:
            world (World): World object containing environment parameters and settings
            config (dict): Configuration dictionary containing visualization settings
                          Must include 'visualizer.colors' with color definitions

        Returns:
            None
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen_width = world.width * 2  # Double width for metrics visualization
        self.screen_height = world.height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Simulation")
        self.font = pygame.font.SysFont(None, 24)
        self.success = False
        self.frozen_data = None
        
        # Initialize color scheme from config
        self.colors = {
            'background': tuple(config['visualizer']['colors']['background']),
            'robot': tuple(config['visualizer']['colors']['robot']),
            'goal': tuple(config['visualizer']['colors']['goal']),
            'obstacle': tuple(config['visualizer']['colors']['obstacle']),
            'success_text': tuple(config['visualizer']['colors']['success_text']),
            'path': tuple(config['visualizer']['colors'].get('path', [0, 0, 255])),
            'reference_trajectory': (0, 255, 0),
            'reference_path': (65, 105, 225),
            'actual_path': (255, 0, 0),
            'reference': (0, 255, 0),
            'actual': (255, 0, 0),
            'inflated_obstacle': (128, 128, 128),
            'alternative_goal': (255, 100, 100)
        }

        self.debug_helper = DebugHelper(debug=True)

    def display_world(self, is_alternative: bool, goal: Tuple[float, float]) -> None:
        """
        Render the complete world state including goals and obstacles.

        Args:
            is_alternative (bool): Flag indicating if alternative goal should be displayed
            goal (Tuple[float, float]): Goal position coordinates (x, y)

        Returns:
            None
        """
        self.display_goal(is_alternative, goal)
        self.display_obstacles()

    def display_goal(self, is_alternative: bool, goal: Tuple[float, float]) -> None:
        """
        Render goal position(s) in the environment.

        Args:
            is_alternative (bool): Flag indicating if alternative goal should be displayed
            goal (Tuple[float, float]): Goal position coordinates (x, y)

        Returns:
            None
        """
        goal = self.world.convert_to_display(goal)
        pygame.draw.circle(self.screen, self.colors['alternative_goal'], goal, 6)
        if is_alternative:
            pygame.draw.circle(self.screen, self.colors['goal'], goal, 6)

    def display_obstacles(self) -> None:
        """
        Render all obstacles in the environment, including inflated safety boundaries.

        Returns:
            None
        """
        # Render inflated obstacles with transparency
        for obstacle in self.world.inflated_obstacles:
            if obstacle.shape == 'circle':
                size = int(obstacle.size * 2)
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*self.colors['inflated_obstacle'], 128), 
                                 (size//2, size//2), int(obstacle.size))
                screen_pos = self.world.convert_to_display(obstacle.position)
                self.screen.blit(surf, (screen_pos[0] - size//2, screen_pos[1] - size//2))
                
            elif obstacle.shape == 'rectangle':
                width, height = obstacle.size
                left = self.world.robot_origin[0] + obstacle.position[0]
                top = self.world.robot_origin[1] - (obstacle.position[1] + height)
                
                surf = pygame.Surface((int(width), int(height)), pygame.SRCALPHA)
                pygame.draw.rect(surf, (*self.colors['inflated_obstacle'], 128), surf.get_rect())
                self.screen.blit(surf, (left, top))

        # Render actual obstacles
        for obstacle in self.world.obstacles:
            if obstacle.shape == 'circle':
                position = self.world.convert_to_display(obstacle.position)
                pygame.draw.circle(self.screen, self.colors['obstacle'], position, int(obstacle.size))
            elif obstacle.shape == 'rectangle':
                width, height = obstacle.size
                left = self.world.robot_origin[0] + obstacle.position[0]
                top = self.world.robot_origin[1] - (obstacle.position[1] + height)
                pygame.draw.rect(self.screen, self.colors['obstacle'], 
                               pygame.Rect(left, top, width, height))

    def display_robot(self, robot: Robot) -> None:
        """
        Render the robot's joints and links.

        Args:
            robot (Robot): Robot object containing current joint positions

        Returns:
            None
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.joint_1_pos())
        j2 = self.world.convert_to_display(robot.joint_2_pos())

        # Render robot structure
        pygame.draw.circle(self.screen, self.colors['robot'], j0, 4)  # Base joint
        pygame.draw.line(self.screen, self.colors['robot'], j0, j1, 2)  # Link 1
        pygame.draw.circle(self.screen, self.colors['robot'], j1, 4)  # Middle joint
        pygame.draw.line(self.screen, self.colors['robot'], j1, j2, 2)  # Link 2
        pygame.draw.circle(self.screen, self.colors['robot'], j2, 4)  # End effector

    def display_paths(self, robot: Robot, controller: 'Controller') -> None:
        """
        Render planned and executed paths of the robot.

        Args:
            robot (Robot): Robot object for forward kinematics calculations
            controller (Controller): Controller object containing path data

        Returns:
            None
        """
        # Render coarse path planning
        if controller.coarse_path:
            reference_points = [self.world.convert_to_display(
                robot.forward_kinematics(state.theta_0, state.theta_1)) 
                for state in controller.coarse_path]
            if len(reference_points) > 1:
                pygame.draw.lines(self.screen, self.colors['reference_path'], 
                                False, reference_points, 2)

        # Render refined trajectory
        if controller.path:
            reference_points = [self.world.convert_to_display(
                robot.forward_kinematics(state.theta_0, state.theta_1)) 
                for state in controller.path]
            if len(reference_points) > 1:
                pygame.draw.lines(self.screen, self.colors['reference_trajectory'], 
                                False, reference_points, 2)

        # Render actual executed path
        if controller.actual_theta_0 and controller.actual_theta_1:
            actual_points = [self.world.convert_to_display(
                robot.forward_kinematics(theta_0, theta_1)) 
                for theta_0, theta_1 in zip(controller.actual_theta_0, controller.actual_theta_1)]
            if len(actual_points) > 1:
                pygame.draw.lines(self.screen, self.colors['actual_path'], 
                                False, actual_points, 2)

    def update_display(self, robot: Robot, success: bool, goal: Tuple[float, float], 
                      is_moving_goal: bool, controller: Optional['Controller'] = None) -> bool:
        """
        Update the complete visualization display.

        Args:
            robot (Robot): Robot object for current state
            success (bool): Flag indicating if goal is reached
            goal (Tuple[float, float]): Current goal position
            is_moving_goal (bool): Flag indicating if goal is currently moving
            controller (Optional[Controller]): Controller object for path visualization

        Returns:
            bool: False if simulation should terminate, True otherwise
        """
        # Handle pygame events
        self.success = success
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False

        self.screen.fill(self.colors['background'])
        
        # Update world visualization
        if controller.is_alternative:
            self.display_world(controller.is_alternative, controller.alternative_goal)
        else:
            self.display_world(controller.is_alternative, goal)
            
        if controller is not None:
            self.display_paths(robot, controller)
        self.display_robot(robot)

        # Display status messages
        if is_moving_goal:
            self._render_text("Planning new Goal!", (self.world.width // 4, 280))
        if success:
            self._render_text("Goal Reached", (self.world.width // 4, 280))

        # Display legends and graphs
        self._display_path_legend()
        if controller is not None:
            self.display_graphs(controller)

        pygame.display.flip()
        return True

    def _render_text(self, text: str, position: Tuple[int, int]) -> None:
        """
        Helper method to render text with consistent styling.

        Args:
            text (str): Text to display
            position (Tuple[int, int]): Center position for text

        Returns:
            None
        """
        text_surface = self.font.render(text, True, self.colors['success_text'])
        text_rect = text_surface.get_rect(center=position)
        self.screen.blit(text_surface, text_rect)

    def _display_path_legend(self) -> None:
        """
        Display legend for different path types.

        Returns:
            None
        """
        legend_x = 50
        legend_y = 10
        legend_spacing = 20

        # Draw legend entries
        legend_items = [
            ("Path", 'reference_path'),
            ("Traj", 'reference_trajectory'),
            ("Ctrl", 'actual_path')
        ]

        for i, (label, color_key) in enumerate(legend_items):
            y_pos = legend_y + i * legend_spacing
            pygame.draw.line(self.screen, self.colors[color_key], 
                           (legend_x, y_pos), (legend_x + 20, y_pos), 2)
            text = self.font.render(label, True, self.colors[color_key])
            self.screen.blit(text, (legend_x + 30, y_pos - 8))

    def display_graphs(self, controller: 'Controller') -> None:
        """
        Display performance metrics graphs. Freezes updates when success is True.

        Args:
            controller (Controller): Controller object containing performance data

        Returns:
            None
        """
        try:
            # Define graph areas
            graph_width = self.screen_width // 2
            graph_height = self.screen_height // 4
            
            # Create graph rectangles
            graphs = [
                ('velocity_0', pygame.Rect(self.screen_width // 2, 0, graph_width, graph_height)),
                ('velocity_1', pygame.Rect(self.screen_width // 2, graph_height, graph_width, graph_height)),
                ('acceleration_0', pygame.Rect(self.screen_width // 2, 2 * graph_height, graph_width, graph_height)),
                ('acceleration_1', pygame.Rect(self.screen_width // 2, 3 * graph_height, graph_width, graph_height))
            ]

            # Draw lighter graph backgrounds
            for _, rect in graphs:
                pygame.draw.rect(self.screen, (100, 100, 100), rect)
                self._draw_grid(rect)

            # If success was just achieved, store the current data
            if self.success and self.frozen_data is None:
                self.frozen_data = {
                    'velocity_0': {
                        'reference': getattr(controller, 'reference_omega_0', []),
                        'actual': getattr(controller, 'actual_omega_0', [])
                    },
                    'velocity_1': {
                        'reference': getattr(controller, 'reference_omega_1', []),
                        'actual': getattr(controller, 'actual_omega_1', [])
                    },
                    'acceleration_0': {
                        'reference': getattr(controller, 'reference_alpha_0', []),
                        'actual': getattr(controller, 'actual_alpha_0', [])
                    },
                    'acceleration_1': {
                        'reference': getattr(controller, 'reference_alpha_1', []),
                        'actual': getattr(controller, 'actual_alpha_1', [])
                    }
                }

            # Use frozen data if success is True, otherwise use current controller data
            data_source = self.frozen_data if self.success else {
                'velocity_0': {
                    'reference': getattr(controller, 'reference_omega_0', []),
                    'actual': getattr(controller, 'actual_omega_0', []),
                    'label': "Velocity Joint 0"
                },
                'velocity_1': {
                    'reference': getattr(controller, 'reference_omega_1', []),
                    'actual': getattr(controller, 'actual_omega_1', []),
                    'label': "Velocity Joint 1"
                },
                'acceleration_0': {
                    'reference': getattr(controller, 'reference_alpha_0', []),
                    'actual': getattr(controller, 'actual_alpha_0', []),
                    'label': "Acceleration Joint 0"
                },
                'acceleration_1': {
                    'reference': getattr(controller, 'reference_alpha_1', []),
                    'actual': getattr(controller, 'actual_alpha_1', []),
                    'label': "Acceleration Joint 1"
                }
            }

            # Plot the data
            for graph_name, rect in graphs:
                if graph_name in data_source:
                    graph_data = data_source[graph_name]
                    ref_data = graph_data['reference']
                    act_data = graph_data['actual']
                    label = graph_name.replace('_', ' ').title()

                    if ref_data is not None and act_data is not None:
                        if len(ref_data) > 0 and len(act_data) > 0:
                            self._plot_data_pair(ref_data, act_data, rect, label)

            # Draw legends
            self.draw_legends()

            # # Add "FROZEN" text when graphs are frozen
            # if self.success:
            #     frozen_text = self.font.render("GRAPHS FROZEN", True, (255, 255, 255))
            #     text_rect = frozen_text.get_rect(center=(self.screen_width * 3/4, 30))
            #     self.screen.blit(frozen_text, text_rect)

        except Exception as e:
            print(f"Error in display_graphs: {str(e)}")

    def _plot_joint_data(self, controller: 'Controller', graphs: List[Tuple[str, pygame.Rect]]) -> None:
        """
        Plot joint velocity and acceleration data from controller.

        Args:
            controller (Controller): Controller containing joint data
            graphs (List[Tuple[str, pygame.Rect]]): List of graph definitions
        """
        try:
            # Data mapping from controller attributes to graph names
            data_mapping = {
                'velocity_0': {
                    'reference': getattr(controller, 'reference_omega_0', []),
                    'actual': getattr(controller, 'actual_omega_0', []),
                    'label': "Velocity Joint 0"
                },
                'velocity_1': {
                    'reference': getattr(controller, 'reference_omega_1', []),
                    'actual': getattr(controller, 'actual_omega_1', []),
                    'label': "Velocity Joint 1"
                },
                'acceleration_0': {
                    'reference': getattr(controller, 'reference_alpha_0', []),
                    'actual': getattr(controller, 'actual_alpha_0', []),
                    'label': "Acceleration Joint 0"
                },
                'acceleration_1': {
                    'reference': getattr(controller, 'reference_alpha_1', []),
                    'actual': getattr(controller, 'actual_alpha_1', []),
                    'label': "Acceleration Joint 1"
                }
            }

            # Plot each graph
            for graph_name, rect in graphs:
                if graph_name in data_mapping:
                    graph_data = data_mapping[graph_name]
                    ref_data = graph_data['reference']
                    act_data = graph_data['actual']
                    label = graph_data['label']

                    # Verify data exists and is valid
                    if ref_data is not None and act_data is not None:
                        if len(ref_data) > 0 and len(act_data) > 0:
                            self._plot_data_pair(ref_data, act_data, rect, label)

        except Exception as e:
            print(f"Error in _plot_joint_data: {str(e)}")
            print(f"Available controller attributes: {dir(controller)}")  # Debug info

    def _plot_data_pair(self, ref_data: List[float], act_data: List[float], 
                       rect: pygame.Rect, label: str) -> None:
        """
        Plot reference and actual data for a single metric.

        Args:
            ref_data (List[float]): Reference data points
            act_data (List[float]): Actual data points
            rect (pygame.Rect): Rectangle defining graph area
            label (str): Graph label

        Returns:
            None
        """
        try:
            if not (ref_data and act_data):
                return

            # Filter out invalid values and convert to float
            ref_valid = [float(v) for v in ref_data if isinstance(v, (int, float)) 
                        and not (np.isnan(v) or np.isinf(v))]
            act_valid = [float(v) for v in act_data if isinstance(v, (int, float)) 
                        and not (np.isnan(v) or np.isinf(v))]
            
            if not (ref_valid and act_valid):
                return

            # Calculate value range with safety checks
            min_val = min(min(act_valid), min(ref_valid))
            max_val = max(max(act_valid), max(ref_valid))
            
            # Ensure non-zero range with sensible defaults
            if max_val == min_val:
                if max_val == 0:
                    max_val = 1.0
                    min_val = -1.0
                else:
                    max_val += abs(max_val) * 0.1
                    min_val -= abs(min_val) * 0.1
            
            padding = (max_val - min_val) * 0.1
            
            # Plot both datasets
            if ref_valid:
                self.plot_graph(ref_valid, self.colors['reference'], rect, 
                              min_val - padding, max_val + padding, label)
            if act_valid:
                self.plot_graph(act_valid, self.colors['actual'], rect, 
                              min_val - padding, max_val + padding)

        except Exception as e:
            print(f"Error in _plot_data_pair: {str(e)}")

    def plot_graph(self, data, color, rect, min_value, max_value, label=None):
        """
        Plot a single data series on a graph.

        Args:
            data (List[float]): Data points to plot
            color (Tuple[int, int, int]): RGB color for the plot
            rect (pygame.Rect): Rectangle defining graph area
            min_value (float): Minimum value for scaling
            max_value (float): Maximum value for scaling
            label (Optional[str]): Graph label

        Returns:
            None
        """
        try:
            if not data or len(data) < 2:
                return

            # Ensure valid scaling
            value_range = max_value - min_value
            if abs(value_range) < 1e-10:  # Avoid division by zero
                value_range = 1.0
                max_value = min_value + 1.0

            x_scale = rect.width / (len(data) - 1)
            y_scale = (rect.height - 20) / value_range  # Leave margin for labels

            # Generate points with boundary checking
            points = []
            for i, value in enumerate(data):
                try:
                    x = rect.left + i * x_scale
                    y = rect.bottom - 10 - ((value - min_value) * y_scale)  # 10px bottom margin
                    # Clamp y value to graph boundaries
                    y = max(rect.top + 10, min(rect.bottom - 10, y))
                    points.append((x, y))
                except (TypeError, ValueError):
                    continue

            # Draw lines with thicker width for better visibility
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)

            # Draw label with better positioning and contrast
            if label:
                label_surf = self.font.render(label, True, (255, 255, 255))
                # Position label in top-left with padding
                self.screen.blit(label_surf, (rect.left + 5, rect.top + 5))

                # Add min/max value labels
                min_label = self.font.render(f"{min_value:.2f}", True, (255, 255, 255))
                max_label = self.font.render(f"{max_value:.2f}", True, (255, 255, 255))
                self.screen.blit(min_label, (rect.left + 5, rect.bottom - 20))
                self.screen.blit(max_label, (rect.left + 5, rect.top + 20))

        except Exception as e:
            print(f"Error in plot_graph: {str(e)}")

    def _draw_grid(self, rect: pygame.Rect) -> None:
        """
        Draw grid lines for better graph readability.
        """
        # Draw horizontal grid lines
        for i in range(4):
            y = rect.top + (rect.height * i) // 3
            pygame.draw.line(self.screen, (150, 150, 150), 
                            (rect.left, y), (rect.right, y), 1)
        
        # Draw vertical grid lines
        for i in range(5):
            x = rect.left + (rect.width * i) // 4
            pygame.draw.line(self.screen, (150, 150, 150), 
                            (x, rect.top), (x, rect.bottom), 1)

    def draw_legends(self):
        """
        Draw legends for all graph metrics.

        Returns:
            None
        """
        legend_x = self.screen_width - 150
        legend_y = 10
        legend_spacing = 20

        # Draw reference legend
        pygame.draw.line(self.screen, self.colors['reference'], 
                        (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        ref_text = self.font.render("Reference", True, self.colors['reference'])
        self.screen.blit(ref_text, (legend_x + 30, legend_y - 8))

        # Draw actual legend
        legend_y += legend_spacing
        pygame.draw.line(self.screen, self.colors['actual'], 
                        (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        act_text = self.font.render("Actual", True, self.colors['actual'])
        self.screen.blit(act_text, (legend_x + 30, legend_y - 8))

    def cleanup(self) -> None:
        """
        Clean up pygame resources before exit.

        Returns:
            None
        """
        pygame.quit()