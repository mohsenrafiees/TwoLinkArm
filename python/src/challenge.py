"""
challenge.py
"""
import time
from typing import List, Tuple, Union
import numpy as np
import pygame


class RobotConstants:
    JOINT_LIMITS = (-2 * np.pi, 2 * np.pi)
    MAX_VELOCITY = 15
    MAX_ACCELERATION = 50
    DT = 0.033  # Time step in seconds
    LINK_1 = 75.0  # Length of first link in pixels
    LINK_2 = 50.0  # Length of second link in pixels


class Robot:
    """
    Represents a two-link robotic arm with constraints on joint angles, velocity, and acceleration.
    """
    def __init__(self) -> None:
        # Initialize angles and their histories
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.theta_0 = 0.0
        self.theta_1 = 0.0

    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self.all_theta_0.append(value)
        self._theta_0 = value
        self._validate_joint_limits(value, 0)
        self._validate_velocity(self.all_theta_0, 0)
        self._validate_acceleration(self.all_theta_0, 0)

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self.all_theta_1.append(value)
        self._theta_1 = value
        self._validate_joint_limits(value, 1)
        self._validate_velocity(self.all_theta_1, 1)
        self._validate_acceleration(self.all_theta_1, 1)

    def _validate_joint_limits(self, theta: float, joint_id: int) -> None:
        if not (RobotConstants.JOINT_LIMITS[0] <= theta <= RobotConstants.JOINT_LIMITS[1]):
            raise ValueError(f"Joint {joint_id} angle {theta} exceeds joint limits.")

    def _validate_velocity(self, all_theta: List[float], joint_id: int) -> None:
        velocity = self._max_velocity(all_theta)
        if velocity > RobotConstants.MAX_VELOCITY:
            raise ValueError(f"Joint {joint_id} velocity {velocity} exceeds limit.")

    def _validate_acceleration(self, all_theta: List[float], joint_id: int) -> None:
        acceleration = self._max_acceleration(all_theta)
        if acceleration > RobotConstants.MAX_ACCELERATION:
            raise ValueError(f"Joint {joint_id} acceleration {acceleration} exceeds limit.")

    def joint_1_pos(self) -> Tuple[float, float]:
        """Compute the position of the first joint."""
        return (
            RobotConstants.LINK_1 * np.cos(self.theta_0),
            RobotConstants.LINK_1 * np.sin(self.theta_0),
        )

    def joint_2_pos(self) -> Tuple[float, float]:
        """Compute the position of the end-effector."""
        return self.forward_kinematics(self.theta_0, self.theta_1)

    @staticmethod
    def forward_kinematics(theta_0: float, theta_1: float) -> Tuple[float, float]:
        """Compute the end-effector position given joint angles."""
        x = (
            RobotConstants.LINK_1 * np.cos(theta_0)
            + RobotConstants.LINK_2 * np.cos(theta_0 + theta_1)
        )
        y = (
            RobotConstants.LINK_1 * np.sin(theta_0)
            + RobotConstants.LINK_2 * np.sin(theta_0 + theta_1)
        )
        return x, y

    @staticmethod
    def inverse_kinematics(x: float, y: float) -> Tuple[float, float]:
        """Compute joint angles from the position of the end-effector."""
        theta_1 = np.arccos(
            (x**2 + y**2 - RobotConstants.LINK_1**2 - RobotConstants.LINK_2**2)
            / (2 * RobotConstants.LINK_1 * RobotConstants.LINK_2)
        )
        theta_0 = np.arctan2(y, x) - np.arctan2(
            RobotConstants.LINK_2 * np.sin(theta_1),
            RobotConstants.LINK_1 + RobotConstants.LINK_2 * np.cos(theta_1),
        )
        return theta_0, theta_1

    @staticmethod
    def _max_velocity(all_theta: List[float]) -> float:
        """Calculate the maximum velocity from joint angle history."""
        return float(max(abs(np.diff(all_theta)) / RobotConstants.DT, default=0))

    @staticmethod
    def _max_acceleration(all_theta: List[float]) -> float:
        """Calculate the maximum acceleration from joint angle history."""
        return float(max(abs(np.diff(np.diff(all_theta))) / RobotConstants.DT**2, default=0))

    @staticmethod
    def min_reachable_radius() -> float:
        """Calculate the minimum reachable radius."""
        return max(RobotConstants.LINK_1 - RobotConstants.LINK_2, 0)

    @staticmethod
    def max_reachable_radius() -> float:
        """Calculate the maximum reachable radius."""
        return RobotConstants.LINK_1 + RobotConstants.LINK_2

# (Continued with refactoring of other classes...)

class World:
    """
    Represents the environment where the robot operates.
    """
    def __init__(self, width: int, height: int, robot_origin: Tuple[int, int], goal: Tuple[int, int]) -> None:
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.goal = goal

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
    COLORS = {
        'BLACK': (0, 0, 0),
        'RED': (255, 0, 0),
        'WHITE': (255, 255, 255)
    }

    def __init__(self, world: World) -> None:
        """
        Initializes the pygame environment and rendering settings.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption("Gherkin Challenge")
        self.font = pygame.font.SysFont(None, 30)

    def display_world(self) -> None:
        """
        Renders the goal position in the world.
        """
        goal = self.world.convert_to_display(self.world.goal)
        pygame.draw.circle(self.screen, self.COLORS['RED'], goal, 6)

    def display_robot(self, robot: Robot) -> None:
        """
        Renders the robot, including joints and links.
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.joint_1_pos())
        j2 = self.world.convert_to_display(robot.joint_2_pos())

        # Render joint 0
        pygame.draw.circle(self.screen, self.COLORS['BLACK'], j0, 4)
        # Render link 1
        pygame.draw.line(self.screen, self.COLORS['BLACK'], j0, j1, 2)
        # Render joint 1
        pygame.draw.circle(self.screen, self.COLORS['BLACK'], j1, 4)
        # Render link 2
        pygame.draw.line(self.screen, self.COLORS['BLACK'], j1, j2, 2)
        # Render joint 2
        pygame.draw.circle(self.screen, self.COLORS['BLACK'], j2, 4)

    def update_display(self, robot: Robot, success: bool) -> bool:
        """
        Updates the display with the latest robot and world states.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(self.COLORS['WHITE'])
        self.display_world()
        self.display_robot(robot)

        if success:
            text = self.font.render("Success!", True, self.COLORS['BLACK'])
            self.screen.blit(text, (10, 10))

        pygame.display.flip()
        return True

    def cleanup(self) -> None:
        """
        Cleans up pygame resources.
        """
        pygame.quit()


class Controller:
    """
    Implements a simple proportional (P) controller for the robot.
    """
    def __init__(self, goal: Tuple[int, int]) -> None:
        self.goal_theta_0, self.goal_theta_1 = Robot.inverse_kinematics(goal[0], goal[1])

    def step(self, robot: Robot) -> Robot:
        """
        Adjusts the robot's joint angles toward the goal using a proportional controller.
        """
        theta_0_error = self.goal_theta_0 - robot.theta_0
        theta_1_error = self.goal_theta_1 - robot.theta_1

        robot.theta_0 += theta_0_error / 10
        robot.theta_1 += theta_1_error / 10

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

    def run(self) -> None:
        """
        Main simulation loop. Steps the controller and updates visualization.
        """
        running = True
        while running:
            self.robot = self.controller.step(self.robot)
            success = self.check_success(self.robot, self.world.goal)
            running = self.visualizer.update_display(self.robot, success)
            time.sleep(RobotConstants.DT)

    @staticmethod
    def check_success(robot: Robot, goal: Tuple[int, int]) -> bool:
        """
        Checks if the robot's end-effector is sufficiently close to the goal.
        """
        return np.allclose(robot.joint_2_pos(), goal, atol=0.25)

    def cleanup(self) -> None:
        """
        Cleans up resources used by the runner.
        """
        self.visualizer.cleanup()


def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[int, int]:
    """
    Generates a random goal within the robot's reachable radius.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(min_radius, max_radius)
    x = int(radius * np.cos(theta))
    y = int(radius * np.sin(theta))
    return x, y


def main() -> None:
    """
    Main entry point for the simulation.
    """
    width, height = 300, 300
    robot_origin = (width // 2, height // 2)
    goal = generate_random_goal(Robot.min_reachable_radius(), Robot.max_reachable_radius())

    robot = Robot()
    controller = Controller(goal)
    world = World(width, height, robot_origin, goal)
    visualizer = Visualizer(world)

    runner = Runner(robot, controller, world, visualizer)

    try:
        runner.run()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Simulation aborted: {e}")
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
