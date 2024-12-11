from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import time
from utils.state import State

if TYPE_CHECKING:
    from enviroment.robot import Robot
    from controllers.controller import Controller

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
            
    def validate_path_limits(self, path: List[State], robot: 'Robot') -> bool:
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
        pass

    
    def within_joint_limits(self, robot: 'Robot', node: Tuple[float, float]) -> bool:
        theta_0, theta_1 = node
        return (robot.constants.JOINT_LIMITS[0] <= theta_0 <= robot.constants.JOINT_LIMITS[1] and
                robot.constants.JOINT_LIMITS[0] <= theta_1 <= robot.constants.JOINT_LIMITS[1])
        pass
    def print_path_stats(self, path: List[State], robot: 'Robot'):
        """Print detailed statistics about the planned path"""
        if not path:
            print("Empty path!")
            return
            
        print("\n=== Planner Output Statistics ===")
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
        pass

    def validate_trajectory_dynamics(self, trajectory: List[State], robot: 'Robot') -> bool:
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
        pass

    def print_trajectory_stats(self, trajectory: List[State], robot: 'Robot'):
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
        pass

    def validate_planner_output(self, path: List[State], start: Tuple[float, float], 
                              goal: Tuple[float, float], robot: 'Robot') -> bool:
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
        pass

    
    def analyze_tracking_performance(self, controller: 'Controller', robot):
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
        pass

    def _estimate_tracking_delay(self, reference, actual, max_delay=20):
        """Estimate tracking delay using cross-correlation"""
        if len(reference) < max_delay * 2:
            return 0
            
        correlations = []
        for delay in range(max_delay):
            correlation = np.corrcoef(reference[delay:], actual[:-delay if delay > 0 else None])[0,1]
            correlations.append(correlation)
        
        return np.argmax(correlations)

    def analyze_control_signals(self, controller: 'Controller', robot):
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
        pass

    def analyze_path_tracking(self, controller: 'Controller', robot):
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
        pass

    def check_dynamic_consistency(self, controller: 'Controller', robot):
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
        pass

    def validate_controller_dynamics(self, controller: 'Controller', robot: 'Robot') -> bool:
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
        pass

    def print_controller_stats(self, controller: 'Controller', robot: 'Robot'):
        """Print detailed statistics about the controller performance"""
        print("\n=== Controller Performance Statistics ===")
        
        # Check if we have enough data points
        if (not hasattr(controller, 'actual_theta_0') or 
            not controller.actual_theta_0 or 
            len(controller.actual_theta_0) < 2):
            print("Insufficient controller data available!")
            return
        pass
            
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

    def print_trajectory_points(self, trajectory: List[State]):
        """Print detailed trajectory point information"""
        print("\n=== Trajectory Points ===")
        print("idx, theta_0, theta_1, omega_0, omega_1, alpha_0, alpha_1")
        
        for i in range(len(trajectory)-1):
            print(f"{i:3d}, {trajectory[i].theta_0:7.3f}, {trajectory[i].theta_1:7.3f}, "
                  f"{trajectory[i].omega_0:7.3f}, {trajectory[i].omega_1:7.3f}, "
                  f"{trajectory[i].alpha_0:7.3f}, {trajectory[i].alpha_1:7.3f}")
        pass

    def print_step_details(self, controller: 'Controller', robot, control_actions):
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
        pass

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
