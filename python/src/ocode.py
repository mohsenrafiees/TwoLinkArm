import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import scipy.integrate
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
        self.tolerance = 1e-6  # Tolerance for binary search
        # Initialize path parameterizer with the path points
        self.path = PathParameterizer(path_points)
        
    def compute_dynamics_at_s(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute M, C, G matrices at given s"""
        point = self.path.get_path_point(s)
        return self.robot.compute_dynamics(point.theta_0, point.theta_1, 0, 0)
        
    def compute_m(self, s: float) -> np.ndarray:
        """Compute m(s) from equation (9.32)"""
        point = self.path.get_path_point(s)
        M, _, _ = self.compute_dynamics_at_s(s)
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        return M @ dtheta
        
    def compute_c(self, s: float) -> np.ndarray:
        """Compute c(s) from equation (9.32)"""
        point = self.path.get_path_point(s)
        M, Gamma, _ = self.compute_dynamics_at_s(s)
        
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        ddtheta = np.array([point.ddtheta_0, point.ddtheta_1])
        
        term1 = M @ ddtheta
        term2 = dtheta.T @ Gamma @ dtheta
        
        return term1 + term2
        
    def compute_g(self, s: float) -> np.ndarray:
        """Get g(s) from dynamics"""
        _, _, G = self.compute_dynamics_at_s(s)
        return G
        
    def compute_limits(self, s: float, s_dot: float) -> Tuple[float, float]:
        """Compute L(s,s_dot) and U(s,s_dot) from equation (9.36)"""
        m = self.compute_m(s)
        c = self.compute_c(s)
        g = self.compute_g(s)
        
        # Initialize with the robot's acceleration limits
        L = -self.robot.constants.MAX_ACCELERATION
        U = self.robot.constants.MAX_ACCELERATION
        
        for i in range(len(m)):
            if abs(m[i]) < self.tolerance:  # Zero inertia case
                # Handle zero inertia point
                continue
                
            c_term = c[i] * s_dot**2
            
            # Get velocity limits for current joint
            vel_max = self.robot.constants.MAX_VELOCITY
            vel_min = self.robot.constants.MIN_VELOCITY
            
            if m[i] > 0:
                Li = (vel_min - c_term - g[i]) / m[i]
                Ui = (vel_max - c_term - g[i]) / m[i]
            else:
                Li = (vel_max - c_term - g[i]) / m[i]
                Ui = (vel_min - c_term - g[i]) / m[i]
                
            L = max(L, Li)
            U = min(U, Ui)
            
            # Ensure we don't exceed robot's acceleration limits
            L = max(L, -self.robot.constants.MAX_ACCELERATION)
            U = min(U, self.robot.constants.MAX_ACCELERATION)
        
        return L, U
        
    def integrate_trajectory(self, s0: float, s_dot0: float, 
                           use_max: bool, t_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate trajectory equation using max or min acceleration"""
        def dynamics(t: float, state: List[float]) -> List[float]:
            s, s_dot = state
            
            # Ensure s is in valid range
            s = np.clip(s, 0, 1)
                
            try:
                L, U = self.compute_limits(s, s_dot)
                s_ddot = U if use_max else L
                
                # Add bounds on acceleration using robot's constants
                s_ddot = np.clip(s_ddot, 
                               -self.robot.constants.MAX_ACCELERATION, 
                               self.robot.constants.MAX_ACCELERATION)
                return [s_dot, s_ddot]
            except Exception as e:
                print(f"Error in dynamics: {e}")
                return [0.0, 0.0]
            
        # Create dense time points for better integration
        t_eval = np.linspace(t_span[0], t_span[1], 100)
            
        solution = scipy.integrate.solve_ivp(
            dynamics,
            t_span,
            [s0, s_dot0],
            method='RK45',
            max_step=0.01,
            rtol=1e-6,
            atol=1e-6,
            t_eval=t_eval  # Force evaluation at these points
        )
        
        if not solution.success:
            print(f"Integration failed: {solution.message}")
            # Return a minimum set of points for interpolation
            return np.array([t_span[0], t_span[1]]), np.array([[s0, s0], [s_dot0, 0.0]])
            
        # Filter out points outside [0,1] range
        mask = (solution.y[0] >= 0) & (solution.y[0] <= 1)
        if not np.any(mask):
            # If no points in range, return boundary points
            return np.array([0, 1]), np.array([[s0, 1.0], [s_dot0, 0.0]])
            
        t_filtered = solution.t[mask]
        y_filtered = solution.y[:, mask]
        
        # Ensure we have at least two points for interpolation
        if len(t_filtered) < 2:
            if s0 <= 0:
                return np.array([0, 0.1]), np.array([[0, 0.1], [s_dot0, s_dot0]])
            elif s0 >= 1:
                return np.array([0.9, 1]), np.array([[0.9, 1], [s_dot0, 0]])
            else:
                return np.array([s0-0.1, s0+0.1]), np.array([[s0-0.1, s0+0.1], [s_dot0, s_dot0]])
                
        return t_filtered, y_filtered
        
    def detect_curve_intersection(self, curve1: np.ndarray, curve2: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect intersection between two phase-plane curves.
        Returns (s, s_dot) at intersection point if found, None otherwise.
        """
        # Ensure we have valid data
        if curve1.size == 0 or curve2.size == 0:
            return None
            
        # Extract time and state arrays
        if len(curve1.shape) == 2 and curve1.shape[0] == 2:
            s1, s_dot1 = curve1[0], curve1[1]
        else:
            return None
            
        if len(curve2.shape) == 2 and curve2.shape[0] == 2:
            s2, s_dot2 = curve2[0], curve2[1]
        else:
            return None
            
        # Sort by s values
        idx1 = np.argsort(s1)
        idx2 = np.argsort(s2)
        s1, s_dot1 = s1[idx1], s_dot1[idx1]
        s2, s_dot2 = s2[idx2], s_dot2[idx2]
        
        # Find overlapping s range
        s_min = max(np.min(s1), np.min(s2))
        s_max = min(np.max(s1), np.max(s2))
        
        if s_min >= s_max:
            return None
            
        try:
            # Interpolate both curves
            f1 = scipy.interpolate.interp1d(s1, s_dot1, bounds_error=False)
            f2 = scipy.interpolate.interp1d(s2, s_dot2, bounds_error=False)
            
            # Create a fine grid of points
            s_points = np.linspace(s_min, s_max, 1000)
            v1 = f1(s_points)
            v2 = f2(s_points)
            
            # Find where curves cross
            mask = ~np.isnan(v1) & ~np.isnan(v2)
            if not np.any(mask):
                return None
                
            s_points = s_points[mask]
            v1 = v1[mask]
            v2 = v2[mask]
            
            diff = v1 - v2
            zero_crossings = np.where(np.diff(np.signbit(diff)))[0]
            
            if len(zero_crossings) == 0:
                return None
                
            # Get the first intersection point
            idx = zero_crossings[0]
            s_intersect = s_points[idx]
            s_dot_intersect = f1(s_intersect)
            
            return (float(s_intersect), float(s_dot_intersect))
            
        except Exception as e:
            print(f"Error in intersection detection: {e}")
            return None

    def check_velocity_limit_violation(self, trajectory: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Check if trajectory violates velocity limits.
        Returns (s, s_dot) at first violation point if found, None otherwise.
        """
        for i in range(len(trajectory[0])):
            s = trajectory[0, i]
            s_dot = trajectory[1, i]
            L, U = self.compute_limits(s, s_dot)
            
            # Check if acceleration limits are violated
            if L > U or s_dot < 0:
                return (s, s_dot)
        
        return None

    def binary_search_velocity(self, s_lim: float, s_dot_lim: float) -> Tuple[float, float]:
        """
        Perform binary search to find tangent point on velocity limit curve.
        Returns (s, s_dot) of tangent point.
        """
        s_dot_low = 0.0
        s_dot_high = s_dot_lim
        best_point = None
        
        while s_dot_high - s_dot_low > self.tolerance:
            s_dot_test = (s_dot_high + s_dot_low) / 2
            
            # Integrate forward from test point
            _, forward = self.integrate_trajectory(s_lim, s_dot_test, False, (0, 10.0))
            violation_point = self.check_velocity_limit_violation(forward)
            
            if violation_point is not None:
                # Trajectory violates limits, try lower velocity
                s_dot_high = s_dot_test
            else:
                # Trajectory respects limits, try higher velocity
                s_dot_low = s_dot_test
                best_point = (s_lim, s_dot_test)
        
        return best_point if best_point is not None else (s_lim, s_dot_low)

    def find_switch_points(self) -> List[Tuple[float, float, TrajectoryType]]:
        """
        Implementation of the algorithm to find switch points.
        Returns list of (s, s_dot, type) tuples for each switch point.
        """
        switches = []
        s_current = 0.0
        s_dot_current = 0.0
        
        print(f"Starting switch point search from ({s_current}, {s_dot_current})")
        
        # Step 2: Integrate backward from (1,0)
        print("Step 2: Integrating backward from (1,0)")
        t_backward, backward = self.integrate_trajectory(1.0, 0.0, False, (0, 10.0))
        if backward.size == 0:
            print("Warning: Backward integration produced no points")
            return []
            
        final_curve = backward
        print(f"Generated {len(final_curve[0])} points on final curve")
        
        iteration = 0
        while s_current < 1.0 and iteration < 20:  # Add iteration limit
            iteration += 1
            print(f"\nIteration {iteration}, current state: ({s_current:.3f}, {s_dot_current:.3f})")
            
            # Step 3: Integrate forward with maximum acceleration
            print("Step 3: Integrating forward")
            t_forward, forward = self.integrate_trajectory(s_current, s_dot_current, True, (0, 10.0))
            print(f"Generated {len(forward[0])} forward points")
            
            # Check for intersection with final curve
            intersection = self.detect_curve_intersection(forward, final_curve)
            
            # Check for velocity limit violation
            violation = self.check_velocity_limit_violation(forward)
            
            if violation is None and intersection is not None:
                print(f"Found intersection at {intersection}")
                s_switch, s_dot_switch = intersection
                switches.append((s_switch, s_dot_switch, TrajectoryType.DECELERATION))
                s_current = s_switch
                s_dot_current = s_dot_switch
            elif violation is not None:
                print(f"Found velocity limit violation at {violation}")
                s_lim, s_dot_lim = violation
                
                # Step 4: Binary search for tangent point
                print("Step 4: Binary searching for tangent point")
                s_tan, s_dot_tan = self.binary_search_velocity(s_lim, s_dot_lim)
                print(f"Found tangent point at ({s_tan:.3f}, {s_dot_tan:.3f})")
                
                # Step 5: Integrate backward from tangent point
                print("Step 5: Integrating backward from tangent point")
                t_back, back = self.integrate_trajectory(s_tan, s_dot_tan, False, (0, 10.0))
                
                # Find intersection with forward curve
                intersection = self.detect_curve_intersection(back, forward)
                if intersection is not None:
                    print(f"Found intersection with forward curve at {intersection}")
                    s_switch, s_dot_switch = intersection
                    switches.append((s_switch, s_dot_switch, TrajectoryType.DECELERATION))
                    switches.append((s_tan, s_dot_tan, TrajectoryType.ACCELERATION))
                    s_current = s_tan
                    s_dot_current = s_dot_tan
                else:
                    print("Warning: Could not find intersection with forward curve")
                    s_current = 1.0  # Exit loop
            else:
                print("No violation or intersection found, ending search")
                break
                
        print(f"Found {len(switches)} switch points")
        return switches

    def compute_optimal_velocity_at_s(self, s: float, switch_points: List[Tuple[float, float, TrajectoryType]]) -> float:
        """
        Compute the optimal velocity at a given s based on switch points.
        """
        # Find the relevant phase between switch points
        current_phase = TrajectoryType.ACCELERATION
        s_prev = 0.0
        s_dot_prev = 0.0
        
        for s_switch, s_dot_switch, switch_type in switch_points:
            if s < s_switch:
                # We're in the current phase
                break
            s_prev = s_switch
            s_dot_prev = s_dot_switch
            current_phase = switch_type
        
        # Integrate from previous switch point to current s
        t_span = (0, 10.0)  # Large enough to reach target s
        _, trajectory = self.integrate_trajectory(s_prev, s_dot_prev, 
                                               current_phase == TrajectoryType.ACCELERATION,
                                               t_span)
        
        # Find s_dot at current s by interpolation
        f = scipy.interpolate.interp1d(trajectory[0], trajectory[1])
        return float(f(s))

    def generate_trajectory(self, path_points: List[State]) -> List[State]:
        """
        Generate time-optimal trajectory from path points.
        Returns list of States with optimal velocities.
        """
        if not path_points:
            return []
        if len(path_points) == 1:
            # Return single point with zero velocity
            return [State(
                theta_0=path_points[0].theta_0,
                theta_1=path_points[0].theta_1,
                omega_0=0.0,
                omega_1=0.0
            )]
            
        try:
            # Initialize path parameterizer
            self.path = PathParameterizer(path_points)
            
            # Find switch points
            switch_points = self.find_switch_points()
            
            # Generate final trajectory
            trajectory_points = []
            
            # Create more dense sampling near switch points
            s_values = []
            base_points = np.linspace(0, 1, max(100, len(path_points) * 2))
            
            for s in base_points:
                s_values.append(s)
                # Add extra points near switch points
                for s_switch, _, _ in switch_points:
                    if abs(s - s_switch) < 0.05:  # Within 5% of switch point
                        # Add points before and after switch
                        s_values.extend([
                            s_switch - 0.01,
                            s_switch - 0.005,
                            s_switch,
                            s_switch + 0.005,
                            s_switch + 0.01
                        ])
            
            s_values = sorted(list(set(np.clip(s_values, 0, 1))))
            
            for s in s_values:
                try:
                    # Get path point at current s
                    point = self.path.get_path_point(s)
                    
                    # Compute optimal velocity (s_dot) at current s
                    s_dot = self.compute_optimal_velocity_at_s(s, switch_points)
                    
                    # Convert path parameter velocity to joint velocities
                    omega_0 = point.dtheta_0 * s_dot
                    omega_1 = point.dtheta_1 * s_dot
                    
                    # Ensure velocities are within bounds
                    omega_0 = np.clip(omega_0, -self.robot.constants.MAX_VELOCITY, 
                                            self.robot.constants.MAX_VELOCITY)
                    omega_1 = np.clip(omega_1, -self.robot.constants.MAX_VELOCITY, 
                                            self.robot.constants.MAX_VELOCITY)
                    
                    # Create state with optimal velocities
                    trajectory_points.append(State(
                        theta_0=point.theta_0,
                        theta_1=point.theta_1,
                        omega_0=omega_0,
                        omega_1=omega_1
                    ))
                except Exception as e:
                    print(f"Error generating trajectory point at s={s}: {e}")
                    continue
            
            if not trajectory_points:
                print("Warning: Failed to generate any valid trajectory points")
                return path_points  # Return original points as fallback
                
            return trajectory_points
            
        except Exception as e:
            print(f"Error in trajectory generation: {e}")
            return path_points  # Return original points as fallback