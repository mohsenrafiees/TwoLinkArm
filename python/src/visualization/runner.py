from typing import Tuple, Optional
import time
import pygame
import numpy as np
from enviroment.robot import Robot
from controller.controller import Controller
from enviroment.world import World
from visualization.visualizer import Visualizer
from utils.debug_helper import DebugHelper
from utils.constants import RobotConstants
from utils.state import State
from utils.utils import generate_random_goal

class Runner:
    """
    Manages simulation execution and coordinates system components.
    
    This class orchestrates the interaction between robot, controller,
    world, and visualization components. It handles:
    - Main simulation loop execution
    - Goal management and updates
    - Path planning coordination
    - Success condition monitoring
    
    Attributes:
        robot (Robot): Robot instance being controlled
        controller (Controller): Motion controller instance
        world (World): Environment representation
        visualizer (Visualizer): Visualization manager
        constants (RobotConstants): Robot physical parameters
        goal (Tuple[float, float]): Current goal position
        is_moving_goal (bool): Whether to simulate moving target
        max_planning_attempts (int): Maximum path planning retries
        planning_attempt_delay (float): Delay between planning attempts (s)
    """
    
    def __init__(self, robot: Robot, controller: Controller, 
                 world: World, visualizer: Visualizer) -> None:
        """
        Initializes simulation runner with system components.
        
        Args:
            robot: Robot instance to control
            controller: Motion controller
            world: Environment representation
            visualizer: Visualization manager
        """
        self.robot = robot
        self.controller = controller
        self.world = world
        self.visualizer = visualizer
        self.debug_helper = DebugHelper(True)
        self.constants = robot.constants
        
        # Goal management
        self.goal = generate_random_goal(
            self.constants.min_reachable_radius(),
            self.constants.max_reachable_radius()
        )
        self.final_velocities = (0.0, 0.0)
        self.goal_change_received = False
        self.new_goal = None
        
        # Planning parameters
        self.max_planning_attempts = 3
        self.planning_attempt_delay = 0.5  # seconds
        self.is_moving_goal = True

    def wait_for_planning(self) -> bool:
        """
        Waits for path planning to complete with timeout and retries.
        
        Returns:
            bool: True if planning succeeded, False if all attempts failed
            
        Note:
            Updates visualization during planning to show progress
        """
        attempts = 0
        while attempts < self.max_planning_attempts:
            # Monitor current planning attempt
            while self.controller.planning_mode:
                if self.controller.is_planning_timeout():
                    self.debug_helper.log_state(f"Planning attempt {attempts + 1} timed out")
                    break
                    
                time.sleep(0.1)
                self.visualizer.update_display(self.robot, False, self.goal, 
                                            True, controller=self.controller)

            # Check for planning success
            if self.controller.path and not self.controller.planning_mode:
                return True

            # Handle retry logic
            attempts += 1
            if attempts < self.max_planning_attempts:
                self.debug_helper.log_state(
                    f"Retrying path planning (attempt {attempts + 1}/{self.max_planning_attempts})")
                time.sleep(self.planning_attempt_delay)
                if not self.controller.set_goal(self.robot, self.new_goal, self.final_velocities):
                    continue

        self.debug_helper.log_state("Failed to find valid path after all attempts")
        return False

    def run(self) -> None:
        """
        Executes main simulation loop.
        
        Handles:
        - Initial path planning
        - Goal updates and replanning
        - Controller step execution
        - Visualization updates
        
        Raises:
            ValueError: If initial path planning fails
        """
        if not self.controller.set_goal(self.robot, self.goal, 
                                      final_velocities=self.final_velocities):
            raise ValueError("Initial path planning failed")

        running = True
        is_moving_goal = self.is_moving_goal
        goal_change_time = time.time() + np.random.uniform(3.0, 4.0)
        last_time = goal_change_time
        
        while running:
            current_time = time.time()
            
            # Handle goal updates for moving target simulation
            if (is_moving_goal and current_time - goal_change_time > 0 
                and not self.goal_change_received):
                self.debug_helper.log_state("Initiating graceful stop")
                self.controller.initiate_graceful_stop(self.robot)
                
                # Generate new target position
                self.new_goal = generate_random_goal(
                    self.constants.min_reachable_radius(),
                    self.constants.max_reachable_radius())
                self.goal_change_received = True
                self.debug_helper.log_state(
                    f"current: {current_time}, goal_change_time: {goal_change_time}")
                    
            # Handle transition to new goal after stop
            if (is_moving_goal and self.goal_change_received 
                and self.controller.stopping_mode):
                    
                if abs(self.robot.omega_0) < 0.01 and abs(self.robot.omega_1) < 0.01:
                    self.debug_helper.log_state("Robot stopped - planning path to new goal")
                    self.goal = self.new_goal
                    self.visualizer.update_display(
                        self.robot, self.check_success(self.robot, self.goal),
                        self.goal, True, controller=self.controller)
                        
                    if not self.controller.set_goal(
                        self.robot, self.goal, final_velocities=self.final_velocities):
                        self.debug_helper.log_state("Failed to set new goal")
                        running = False
                        break

                    self.visualizer.update_display(
                        self.robot, self.check_success(self.robot, self.goal),
                        self.goal, False, controller=self.controller)
                    self.debug_helper.log_state("Successfully planned path to new goal")
                    
                    # Reset for next goal change
                    self.goal_change_received = False
                    goal_change_time = time.time() + np.random.uniform(3.0, 4.0)
            
            # Execute normal control loop
            if not self.controller.planning_mode:
                self.robot = self.controller.step(self.robot)
                success = self.check_success(self.robot, self.goal)
                self.visualizer.update_display(
                    self.robot, success, self.goal, False, controller=self.controller)

                # exit if goal succedded and we are not in moaving_goal_mode
                if success and not self.is_moving_goal:
                    running = False

            
            time.sleep(self.constants.DT)

        # Handle simulation pause/exit
        pause = True
        while pause:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pause = False
            pygame.display.flip()
            pygame.time.wait(100)

    def check_success(self, robot: Robot, goal: Tuple[float, float]) -> bool:
        """
        Determines if robot has reached goal position.
        
        Args:
            robot: Robot instance to check
            goal: Target position to reach
            
        Returns:
            bool: True if robot is sufficiently close to goal
            
        Note:
            Uses position threshold of 0.5 units for success criterion
        """
        position_threshold = self.controller.goal_distance + 0.1 # 0.1 for buffer 
        velocity_threshold = 0.005
        
        current_pos = robot.joint_2_pos()
        position_error = np.hypot(
            current_pos[0] - goal[0], 
            current_pos[1] - goal[1]
        )
        
        position_close = position_error <= position_threshold
        velocity_close = (abs(robot.omega_0) <= velocity_threshold and 
                         abs(robot.omega_1) <= velocity_threshold)
        
        return position_close and velocity_threshold 

    def cleanup(self) -> None:
        """
        Releases resources used by the simulation.
        
        Ensures proper cleanup of visualization resources.
        """
        self.visualizer.cleanup()