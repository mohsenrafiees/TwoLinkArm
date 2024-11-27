import time
from typing import List, Tuple, Union
import numpy as np
import pygame
import json


class RobotConstants:
    def __init__(self, config):
        self.JOINT_LIMITS = tuple(config['joint_limits'])
        self.MAX_VELOCITY = config['max_velocity']
        self.MAX_ACCELERATION = config['max_acceleration']
        self.DT = config['dt']
        self.LINK_1 = config['link_lengths'][0]
        self.LINK_2 = config['link_lengths'][1]

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
        self._theta_0 = 0.0
        self._theta_1 = 0.0

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

    def joint_1_pos(self) -> Tuple[float, float]:
        """Compute the position of the first joint."""
        return (
            self.constants.LINK_1 * np.cos(self.theta_0),
            self.constants.LINK_1 * np.sin(self.theta_0),
        )

    def joint_2_pos(self) -> Tuple[float, float]:
        """Compute the position of the end-effector."""
        return self.forward_kinematics(self.theta_0, self.theta_1)

    def compute_link_segments(self, theta_0: float, theta_1: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Compute line segments representing the robot's links."""
        j0 = (0.0, 0.0)
        j1 = (
            self.constants.LINK_1 * np.cos(theta_0),
            self.constants.LINK_1 * np.sin(theta_0),
        )
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
            'success_text': tuple(config['visualizer']['colors']['success_text'])
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
            ox, oy = obstacle.position
            if obstacle.shape == 'circle':
                position = self.world.convert_to_display(obstacle.position)
                pygame.draw.circle(self.screen, self.colors['obstacle'], position, int(obstacle.size))
            elif obstacle.shape == 'rectangle':
                width, height = obstacle.size
                # Compute display coordinates for the top-left corner
                left = self.world.robot_origin[0] + ox
                top = self.world.robot_origin[1] - (oy + height)
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

    def update_display(self, robot: Robot, success: bool, goal: Tuple[float, float]) -> bool:
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


class Controller:
    """
    Implements a simple proportional (P) controller for the robot with collision checking for the entire robot body.
    """

    def __init__(self, constants: RobotConstants, world: World) -> None:
        self.constants = constants
        self.world = world

    def step(self, robot: Robot, goal: Tuple[float, float]) -> Robot:
        """
        Adjusts the robot's joint angles toward the goal using a proportional controller with collision avoidance.
        """
        goal_theta_0, goal_theta_1 = robot.inverse_kinematics(goal[0], goal[1])

        theta_0_error = goal_theta_0 - robot.theta_0
        theta_1_error = goal_theta_1 - robot.theta_1

        # Compute proposed next angles
        delta_theta_0 = theta_0_error / 10
        delta_theta_1 = theta_1_error / 10
        next_theta_0 = robot.theta_0 + delta_theta_0
        next_theta_1 = robot.theta_1 + delta_theta_1

        # Predict next link segments
        next_link_segments = robot.compute_link_segments(next_theta_0, next_theta_1)

        if self.check_collision(next_link_segments):
            # If collision detected, stop movement or adjust path
            pass  # For simplicity, we stop updating the angles
        else:
            robot.theta_0 = next_theta_0
            robot.theta_1 = next_theta_1

        return robot

    def check_collision(self, link_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        for obstacle in self.world.obstacles:
            for segment in link_segments:
                if self.is_colliding(segment, obstacle):
                    return True
        return False

    def is_colliding(self, segment: Tuple[Tuple[float, float], Tuple[float, float]], obstacle: Obstacle) -> bool:
        p1, p2 = segment
        if obstacle.shape == 'circle':
            return self.line_circle_collision(p1, p2, obstacle)
        elif obstacle.shape == 'rectangle':
            return self.line_rectangle_collision(p1, p2, obstacle)
        # Additional shapes can be added here
        return False

    def line_circle_collision(self, p1: Tuple[float, float], p2: Tuple[float, float], obstacle: Obstacle) -> bool:
        # Line segment p1-p2 and circle at obstacle.position with radius obstacle.size
        cx, cy = obstacle.position
        r = obstacle.size
        # Algorithm from: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        # Convert to vector form
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        fx = p1[0] - cx
        fy = p1[1] - cy

        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            # No intersection
            return False
        else:
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant)/(2*a)
            t2 = (-b + discriminant)/(2*a)
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
            return False

    def line_rectangle_collision(self, p1: Tuple[float, float], p2: Tuple[float, float], obstacle: Obstacle) -> bool:
        # Line segment p1-p2 and rectangle defined by obstacle.position and obstacle.size
        rx, ry = obstacle.position
        rw, rh = obstacle.size
        # Edges of the rectangle
        left = rx
        right = rx + rw
        top = ry + rh
        bottom = ry

        # Check collision with each side of the rectangle
        # Top edge
        if self.line_line_collision(p1, p2, (left, top), (right, top)):
            return True
        # Bottom edge
        if self.line_line_collision(p1, p2, (left, bottom), (right, bottom)):
            return True
        # Left edge
        if self.line_line_collision(p1, p2, (left, bottom), (left, top)):
            return True
        # Right edge
        if self.line_line_collision(p1, p2, (right, bottom), (right, top)):
            return True
        return False

    def line_line_collision(self, p1: Tuple[float, float], p2: Tuple[float, float],
                            q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        # Check if line segments p1-p2 and q1-q2 intersect
        # Algorithm from: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


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
        while running:
            self.robot = self.controller.step(self.robot, self.goal)
            success = self.check_success(self.robot, self.goal)
            running = self.visualizer.update_display(self.robot, success, self.goal)
            if success:
                # Generate a new random goal
                self.goal = generate_random_goal(
                    self.constants.min_reachable_radius(),
                    self.constants.max_reachable_radius()
                )
            time.sleep(self.constants.DT)

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


def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[float, float]:
    """
    Generates a random goal within the robot's reachable radius.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(min_radius, max_radius)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def generate_random_rectangle_obstacle(workspace_size: float, min_distance: float) -> Obstacle:
    """
    Generates a random rectangle obstacle within the workspace, ensuring it's far enough from the robot's base.
    """
    max_attempts = 100  # Prevent infinite loops
    for _ in range(max_attempts):
        # Define maximum size for the rectangle
        max_width = workspace_size / 4
        max_height = workspace_size / 4

        # Random width and height
        width = np.random.uniform(workspace_size / 10, max_width)
        height = np.random.uniform(workspace_size / 10, max_height)

        # Random position within workspace
        x_min = -workspace_size / 2 + width
        x_max = workspace_size / 2 - width
        y_min = -workspace_size / 2 + height
        y_max = workspace_size / 2 - height

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # Check distance from robot's base at (0, 0)
        obstacle_center_x = x + width / 2
        obstacle_center_y = y + height / 2
        distance = np.hypot(obstacle_center_x, obstacle_center_y)

        # Consider half of the obstacle's diagonal as its radius
        obstacle_radius = np.hypot(width / 2, height / 2)

        if distance - obstacle_radius >= min_distance:
            return Obstacle('rectangle', (x, y), (width, height))

    # If unable to find a suitable position, raise an exception
    raise ValueError("Unable to generate obstacle far enough from the robot's base.")


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

    # Initialize controller
    controller = Controller(robot_constants, world)

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
