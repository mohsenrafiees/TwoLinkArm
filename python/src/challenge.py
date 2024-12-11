import time
import json
import pygame
import numpy as np
from typing import List, Tuple, Optional

from enviroment.robot import Robot
from enviroment.world import World
from enviroment.obstacles import Obstacle
from controller.controller import Controller
from visualization.visualizer import Visualizer
from visualization.runner import Runner
from utils.constants import RobotConstants
from utils.debug_helper import DebugHelper
from planner.sbpl_planner import SBPLPlanner
from utils.state import State
from utils.path_point import PathPoint
from utils.utils import generate_random_circle_obstacle

def main() -> None:
    """
    Main entry point for the simulation.
    """
    # Load configuration
    with open('Config/config.json', 'r') as f:
        config = json.load(f)
    # Add gravity consideration to config
    consider_gravity = True
    config['robot']['consider_gravity'] = consider_gravity
    # Initialize robot constants
    robot_constants = RobotConstants(config['robot'])
    # Initialize obstacles
    obstacles = [Obstacle(**obs) for obs in config['world'].get('obstacles', [])]
    # circle_obstacle = generate_random_circle_obstacle(config['world']['width'], 30.0)
    # obstacles.append(circle_obstacle)
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
    planner = SBPLPlanner(robot, world, config)

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

if __name__ == "__main__":
    main()