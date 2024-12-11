from typing import List, Tuple, Optional
import numpy as np
from enum import Enum
import scipy.integrate
import scipy.interpolate
from enviroment.robot import Robot
from utils.state import State
from utils.path_point import PathPoint
from utils.debug_helper import DebugHelper
from trajectory.parameterizer import PathParameterizer

class TrajectoryType(Enum):
    """Defines the types of trajectory segments for time-optimal path following."""
    ACCELERATION = 'acceleration'
    DECELERATION = 'deceleration'

class TimeOptimalTrajectoryGenerator:
    """
    Generates time-optimal trajectories for a two-link robotic manipulator while respecting
    dynamic constraints and velocity limits. Formulation can befound in  9.4 Time-Optimal Time Scaling in following book
    https://hades.mech.northwestern.edu/index.php/Modern_Robotics
    """
    
    def __init__(self, robot: Robot, path_points: List[State]):
        """
        Initialize the trajectory generator with robot model and path points.

        Args:
            robot: Robot model containing dynamic properties and constraints
            path_points: List of states defining the geometric path to follow
        """
        self.robot = robot
        self.tolerance = 1e-4  # Numerical tolerance for optimization
        self.debug = DebugHelper(debug=True)
        self.debug.print_trajectory_header()
        self.path = PathParameterizer(path_points)
        self.cached_dynamics = {}
        
    def compute_dynamics_at_s(self, s: float, omega_0: float = 0.0, omega_1: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute robot dynamics matrices at a given path parameter with caching.

        Args:
            s: Path parameter value [0,1]
            omega_0: Angular velocity of first joint (rad/s)
            omega_1: Angular velocity of second joint (rad/s)

        Returns:
            Tuple containing:
            - M: Mass matrix
            - C: Coriolis matrix 
            - G: Gravity vector
        """
        if len(self.cached_dynamics) > 1000000:
            self._prune_cache()

        cache_key = (round(s, 4), round(omega_0, 4), round(omega_1, 4))
        if cache_key in self.cached_dynamics:
            return self.cached_dynamics[cache_key]

        point = self.path.get_path_point(s)
        M, C, G = self.robot.compute_dynamics(point.theta_0, point.theta_1, omega_0, omega_1)
        self.cached_dynamics[cache_key] = (M, C, G)
        return M, C, G
        
    def _prune_cache(self):
        """
        Prunes the dynamics cache to prevent memory overflow while keeping the most relevant entries.
        Uses a scoring system based on path parameter distance from endpoints and velocity magnitudes.
        """
        MAX_CACHE_SIZE = 1000000
        TARGET_SIZE = int(MAX_CACHE_SIZE * 0.75)

        if len(self.cached_dynamics) <= TARGET_SIZE:
            return

        sorted_entries = sorted(
            self.cached_dynamics.items(),
            key=lambda x: self._compute_cache_score(x[0]),
            reverse=True
        )
        self.cached_dynamics = dict(sorted_entries[:TARGET_SIZE])

    def _compute_cache_score(self, cache_key: Tuple[float, float, float]) -> float:
        """
        Compute relevance score for cache entries to determine which to keep during pruning.

        Args:
            cache_key: Tuple of (s, omega_0, omega_1) identifying the cached entry

        Returns:
            float: Relevance score where higher values indicate more important entries
        """
        s, omega_0, omega_1 = cache_key
        s_weight = 1.0
        velocity_weight = 0.5
        
        endpoint_distance = min(abs(s), abs(1.0 - s))
        s_score = 1.0 - endpoint_distance
        
        velocity_magnitude = np.sqrt(omega_0**2 + omega_1**2)
        velocity_score = 1.0 / (1.0 + velocity_magnitude)
        
        return (s_weight * s_score + velocity_weight * velocity_score) / (s_weight + velocity_weight)

    def compute_christoffel_symbols(self, theta_0: float, theta_1: float) -> np.ndarray:
        """
        Compute Christoffel symbols for the robot's dynamics using numerical differentiation.

        Args:
            theta_0: First joint angle (rad)
            theta_1: Second joint angle (rad)

        Returns:
            3D array of Christoffel symbols Γᵏᵢⱼ
        """
        h = 1e-6  # Step size for numerical differentiation
        M, _, _ = self.robot.compute_dynamics(theta_0, theta_1, 0, 0)
        
        # Compute partial derivatives of mass matrix
        dM_dtheta0 = (self.robot.compute_dynamics(theta_0 + h, theta_1, 0, 0)[0] - 
                      self.robot.compute_dynamics(theta_0 - h, theta_1, 0, 0)[0]) / (2*h)
        dM_dtheta1 = (self.robot.compute_dynamics(theta_0, theta_1 + h, 0, 0)[0] - 
                      self.robot.compute_dynamics(theta_0, theta_1 - h, 0, 0)[0]) / (2*h)
        
        # Initialize Christoffel symbols tensor
        Gamma = np.zeros((2, 2, 2))
        M_inv = np.linalg.inv(M)
        
        # Compute symbols using the standard formula
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        dM_dq = dM_dtheta0 if l == 0 else dM_dtheta1
                        term1 = dM_dq[i, l]
                        term2 = dM_dq[j, l]
                        term3 = (dM_dtheta0 if j == 0 else dM_dtheta1)[i, j]
                        Gamma[k, i, j] += 0.5 * M_inv[k, l] * (term1 + term2 - term3)
        
        return Gamma

    def compute_m(self, s: float) -> np.ndarray:
        """
        Compute the mass-dependent term m(s) for phase plane analysis.

        Args:
            s: Path parameter value [0,1]

        Returns:
            Mass-dependent vector m(s)
        """
        point = self.path.get_path_point(s)
        M, _, _ = self.compute_dynamics_at_s(s)
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        return M @ dtheta

    def compute_g(self, s: float) -> np.ndarray:
        """
        Get gravity vector at given path parameter.

        Args:
            s: Path parameter value [0,1]

        Returns:
            Gravity force vector
        """
        _, _, G = self.compute_dynamics_at_s(s)
        return G

    def compute_c(self, s: float, s_dot: float) -> np.ndarray:
        """
        Compute the Coriolis and centrifugal terms for phase plane analysis.

        Args:
            s: Path parameter value [0,1]
            s_dot: Path parameter velocity

        Returns:
            Coriolis and centrifugal force vector
        """
        point = self.path.get_path_point(s)
        M, _, _ = self.compute_dynamics_at_s(s)
        
        dtheta = np.array([point.dtheta_0, point.dtheta_1])
        ddtheta = np.array([point.ddtheta_0, point.ddtheta_1])
        
        # Compute using Christoffel symbols for accuracy
        Gamma = self.compute_christoffel_symbols(point.theta_0, point.theta_1)
        term1 = M @ ddtheta
        term2 = np.zeros(2)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    term2[i] += Gamma[i, j, k] * dtheta[j] * dtheta[k]

        return term1 + term2

    
    def compute_limits(self, s: float, s_dot: float) -> Tuple[float, float]:
        """
        Compute acceleration bounds L(s,s_dot) and U(s,s_dot) for phase plane analysis.

        Args:
            s: Path parameter value [0,1]
            s_dot: Path parameter velocity

        Returns:
            Tuple of (L, U) representing lower and upper acceleration bounds
        """
        m = self.compute_m(s)
        c = self.compute_c(s, s_dot)
        g = self.compute_g(s)
        
        L = -self.robot.constants.MAX_ACCELERATION
        U = self.robot.constants.MAX_ACCELERATION
        
        for i in range(len(m)):
            if abs(m[i]) < self.tolerance:
                continue
                
            c_term = c[i] * s_dot**2
            accel_max = self.robot.constants.MAX_ACCELERATION
            accel_min = -self.robot.constants.MAX_ACCELERATION
            
            if m[i] > 0:
                Li = (accel_min - c_term - g[i]) / m[i]
                Ui = (accel_max - c_term - g[i]) / m[i]
            else:
                Li = (accel_max - c_term - g[i]) / m[i]
                Ui = (accel_min - c_term - g[i]) / m[i]
                
            L = max(L, Li)
            U = min(U, Ui)
        
        return (max(L, -self.robot.constants.MAX_ACCELERATION),
                min(U, self.robot.constants.MAX_ACCELERATION))
    
    def find_switch_points(self) -> List[Tuple[float, float, TrajectoryType]]:
        """
        Find optimal switching points between acceleration and deceleration phases.
        Uses adaptive step size and binary search for efficient computation.

        Returns:
            List of (s, s_dot, trajectory_type) tuples defining switching points
        """

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
        t_backward, backward = self.integrate_trajectory(1.0, 0.0, False, (0, 30.0))
        
        if backward.size == 0:
            self.debug.log_state("Warning: Backward integration produced no points")
            return []
            
        final_curve = backward
        self.debug.log_state(f"Generated {len(final_curve[0])} points on final curve")
        
        # Use adaptive step size
        iteration = 0
        max_iterations = 1000# Reduced max iterations
        min_accaptable_s = 0.98
        
        while s_current < 1.0 and (iteration < max_iterations  and s_current > min_accaptable_s):
            iteration += 1
            self.debug.log_state(f"\n=== Iteration {iteration} ===")
            self.debug.log_state(f"Current state: (s={s_current:.3f}, s_dot={s_dot_current:.3f})")
            self.debug.log_state(f"Current step size: {step_size:.3f}")
            
            # Forward integration
            self.debug.log_state("\nForward integration with maximum acceleration")
            t_forward, forward = self.integrate_trajectory(s_current, s_dot_current, True, (0, 30.0))
            
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
                    s_dot_current = 0.0
                
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
                        step_size = max(min_step, step_size * reduction_factor)
                    else:
                        self.debug.log_state("Switch point too close to current point, advancing")
                        s_current += step_size
                        s_dot_current = 0.0
                else:
                    self.debug.log_state("No intersection found with forward curve, advancing")
                    s_current += step_size
                    s_dot_current = 0.0
                    step_size = min(step_size, step_size * growth_factor)  # Increase step size when no intersection
            else:
                self.debug.log_state("No violation or intersection found, ending search")
                break
                
        self.debug.log_state(f"\nSwitch point search complete. Found {len(switches)} switch points:")
        for i, (s, s_dot, switch_type) in enumerate(switches):
            self.debug.log_state(f"Switch point {i+1}: (s={s:.3f}, s_dot={s_dot:.3f}, type={switch_type})")
            
        return switches

    def integrate_trajectory(self, s0: float, s_dot0: float, use_max: bool, 
                           t_span: Tuple[float, float], tol = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate trajectory in phase plane using adaptive step size control.

        Args:
            s0: Initial path parameter
            s_dot0: Initial velocity
            use_max: If True, use maximum acceleration; if False, use minimum
            t_span: Time interval for integration (t_start, t_end)

        Returns:
            Tuple of (time_points, states) where states contains [s, s_dot] trajectories
        """
        def dynamics(t: float, state: List[float]) -> List[float]:
            s, s_dot = state
            s = np.clip(s, 0, 1)
            try:
                L, U = self.compute_limits(s, s_dot)
                s_ddot = U if use_max else L
                s_ddot = np.clip(s_ddot, -self.robot.constants.MAX_ACCELERATION, 
                                        self.robot.constants.MAX_ACCELERATION)
                return [s_dot, s_ddot]
            except Exception as e:
                self.debug.log_state(f"Error in dynamics: {e}")
                return [0.0, 0.0]

        # Adaptive step size based on path curvature
        def get_timestep(s):
            point = self.path.get_path_point(s)
            curvature = abs(point.ddtheta_0) + abs(point.ddtheta_1)
            return min(0.1, max(0.05, 0.1 / (1 + curvature)))
        
        solution = scipy.integrate.solve_ivp(
            dynamics,
            t_span,
            [s0, s_dot0],
            method='RK45',
            max_step=get_timestep(s0),
            rtol=tol,
            atol=tol,
            dense_output=True
        )
        
        if not solution.success:
            return np.array([t_span[0], t_span[1]]), np.array([[s0, s0], [s_dot0, 0.0]])
        
        # Filter valid points with preference for reaching s=1
        mask = (solution.y[0] >= 0) & (solution.y[0] <= 1)
        if not np.any(mask):
            return np.array([0, 1]), np.array([[s0, 1.0], [s_dot0, 0.0]])
    
        # Find the point that gets closest to s=1
        valid_s = solution.y[0][mask]
        if len(valid_s) > 0:
            max_s_idx = np.argmax(valid_s)
            # If we're reasonably close to 1, truncate the trajectory there
            if valid_s[max_s_idx] > 0.95:
                mask = mask & (np.arange(len(mask)) <= np.where(mask)[0][max_s_idx])
    
        return solution.t[mask], solution.y[:, mask]

    def binary_search_velocity(self, s_lim: float, s_dot_lim: float) -> Tuple[float, float]:
        """
        Performs an optimized binary search to find maximum feasible velocity at given path parameter.
        
        Args:
            s_lim (float): Path parameter value to search at
            s_dot_lim (float): Upper bound for velocity search
            
        Returns:
            Tuple[float, float]: Optimal (s, s_dot) pair that satisfies velocity constraints
            
        Note:
            Uses early convergence optimization with relative tolerance checking
        """
        self.debug.log_state(f"\nStarting binary search for velocity at s={s_lim:.3f}, s_dot_lim={s_dot_lim:.3f}")
        
        s_dot_low = 0.0  # Initialize lower bound
        s_dot_high = s_dot_lim  # Initialize upper bound
        best_point = None
        
        # Search parameters
        max_iterations = 20
        rel_tolerance = 3e-3
        last_s_dot_test = s_dot_high

        for iteration in range(max_iterations):
            s_dot_test = (s_dot_high + s_dot_low) / 2
            self.debug.log_state(f"\nBinary search iteration {iteration + 1}:")
            self.debug.log_state(f"  Testing s_dot={s_dot_test:.3f}")
            
            _, forward = self.integrate_trajectory(s_lim, s_dot_test, False, (0, 10.0))
            violation_point = self.check_velocity_limit_violation(forward)
            
            # Update bounds based on constraint violation
            if violation_point is not None:
                self.debug.log_state(f"  Trajectory violates limits at {violation_point}")
                s_dot_high = s_dot_test
            else:
                self.debug.log_state(f"  Trajectory respects limits")
                s_dot_low = s_dot_test
                best_point = (s_lim, s_dot_test)
                
            self.debug.log_state(f"  Updated bounds: [{s_dot_low:.3f}, {s_dot_high:.3f}]")
            
            if abs(s_dot_test - last_s_dot_test) < rel_tolerance:
                self.debug.log_state("  Converged within relative tolerance")
                break
            last_s_dot_test = s_dot_test
                
        final_point = best_point if best_point is not None else (s_lim, s_dot_low)
        self.debug.log_state(f"Binary search complete. Found point: (s={final_point[0]:.3f}, s_dot={final_point[1]:.3f})")
        return final_point

    def detect_curve_intersection(self, curve1: np.ndarray, curve2: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detects intersection point between two phase-plane curves.
        
        Args:
            curve1 (np.ndarray): First curve data in shape (2, N) where rows are s and s_dot values
            curve2 (np.ndarray): Second curve data in shape (2, N) where rows are s and s_dot values
            
        Returns:
            Optional[Tuple[float, float]]: (s, s_dot) at intersection if found, None otherwise
            
        Note:
            Uses interpolation to find precise intersection point in overlapping s-range
        """
        self.debug.log_state("\nSearching for curve intersection")
        
        # Validate input data
        if curve1.size == 0 or curve2.size == 0:
            self.debug.log_state("Invalid curve data")
            return None
            
        # Extract and validate curve components
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
            
        # Sort curves by s values for interpolation
        idx1 = np.argsort(s1)
        idx2 = np.argsort(s2)
        s1, s_dot1 = s1[idx1], s_dot1[idx1]
        s2, s_dot2 = s2[idx2], s_dot2[idx2]
        
        # Find valid intersection range
        s_min = max(np.min(s1), np.min(s2))
        s_max = min(np.max(s1), np.max(s2))
        
        self.debug.log_state(f"Overlapping s range: [{s_min:.3f}, {s_max:.3f}]")
        
        if s_min >= s_max:
            self.debug.log_state("No overlapping range found")
            return None
            
        try:
            # Create interpolation functions
            f1 = scipy.interpolate.interp1d(s1, s_dot1, bounds_error=False)
            f2 = scipy.interpolate.interp1d(s2, s_dot2, bounds_error=False)
            
            # Evaluate on fine grid
            s_points = np.linspace(s_min, s_max, 100)
            v1 = f1(s_points)
            v2 = f2(s_points)
            
            # Find valid intersection points
            mask = ~np.isnan(v1) & ~np.isnan(v2)
            if not np.any(mask):
                self.debug.log_state("No valid intersection points found")
                return None
                
            s_points = s_points[mask]
            v1 = v1[mask]
            v2 = v2[mask]
            
            # Detect zero crossings
            diff = v1 - v2
            zero_crossings = np.where(np.diff(np.signbit(diff)))[0]
            
            if len(zero_crossings) == 0:
                self.debug.log_state("No zero crossings found")
                return None
                
            # Return first intersection point
            idx = zero_crossings[0]
            s_intersect = s_points[idx]
            s_dot_intersect = f1(s_intersect)
            
            self.debug.log_state(f"Found intersection at (s={s_intersect:.3f}, s_dot={s_dot_intersect:.3f})")
            return (float(s_intersect), float(s_dot_intersect))
            
        except Exception as e:
            self.debug.log_state(f"Error in intersection detection: {e}")
            return None

    def check_velocity_limit_violation(self, trajectory: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Checks if a trajectory violates velocity constraints at any point.
        
        Args:
            trajectory (np.ndarray): Trajectory data in shape (2, N) containing s and s_dot values
            
        Returns:
            Optional[Tuple[float, float]]: (s, s_dot) at first violation point if found, None if no violations
        """
        self.debug.log_state("\nChecking for velocity limit violations")
        
        for i in range(len(trajectory[0])):
            s = trajectory[0, i]
            s_dot = trajectory[1, i]
            L, U = self.compute_limits(s, s_dot)
            
            if L > U or s_dot < 0:
                self.debug.log_state(f"Violation found at (s={s:.3f}, s_dot={s_dot:.3f})")
                self.debug.log_state(f"  Limits at violation: L={L:.3f}, U={U:.3f}")
                return (s, s_dot)
        
        self.debug.log_state("No velocity limit violations found")
        return None

    def compute_optimal_velocity_at_s(self, s: float, switch_points: List[Tuple[float, float, TrajectoryType]]) -> float:
        """
        Computes optimal velocity at a given path parameter value.
        
        Args:
            s (float): Path parameter value to compute velocity at
            switch_points (List[Tuple[float, float, TrajectoryType]]): List of (s, s_dot, type) trajectory switch points
            
        Returns:
            float: Optimal velocity value at given s
            
        Note:
            Integrates trajectory from previous switch point to compute velocity
        """
        self.debug.log_state(f"\nComputing optimal velocity at s={s:.3f}")
        
        # Determine current phase from switch points
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
        
        # Integrate trajectory
        t_span = (0, 10.0)
        _, trajectory = self.integrate_trajectory(s_prev, s_dot_prev, 
                                               current_phase == TrajectoryType.ACCELERATION,
                                               t_span)
        
        try:
            # Determine valid interpolation range
            s_min, s_max = np.min(trajectory[0]), np.max(trajectory[0])
            self.debug.log_state(f"Valid interpolation range: [{s_min:.3f}, {s_max:.3f}]")
            
            s_clipped = np.clip(s, s_min, s_max)
            if s != s_clipped:
                self.debug.log_state(f"Warning: s={s:.3f} outside valid range, clipped to {s_clipped:.3f}")
            
            # Interpolate velocity
            f = scipy.interpolate.interp1d(trajectory[0], trajectory[1], bounds_error=False, fill_value="extrapolate")
            s_dot = float(f(s_clipped))
            
            if abs(s_clipped - s_max) < self.tolerance:
                self.debug.log_state("At maximum valid s, returning zero velocity")
                s_dot = 0.0
                
            self.debug.log_state(f"Computed optimal velocity: s_dot={s_dot:.3f}")
            return s_dot
            
        except Exception as e:
            self.debug.log_state(f"Error in velocity computation: {e}")
            return 0.0

    def calculate_path_length(self, num_samples: int = 100) -> float:
        """
        Calculates the spatial path length by sampling points and computing
        cumulative Euclidean distances.
        
        Args:
            num_samples (int): Number of samples to use for length calculation
            
        Returns:
            float: Total path length in meters
        """
        path_length = 0.0
        prev_pos = None
        
        # Sample points along the path
        for i in range(num_samples):
            s = i / (num_samples - 1)
            path_point = self.path.get_path_point(s)
            
            # Get cartesian coordinates using forward kinematics
            current_pos = self.robot.forward_kinematics(
                path_point.theta_0, 
                path_point.theta_1
            )
            
            # Calculate distance from previous point
            if prev_pos is not None:
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                segment_length = np.sqrt(dx**2 + dy**2)
                path_length += segment_length
                
            prev_pos = current_pos
            
        return path_length

    def generate_trajectory(self, path_points: List[State]) -> List[State]:
        """
        Generates time-optimal trajectory from given path points.
        
        Args:
            path_points (List[State]): List of path points containing joint positions
            
        Returns:
            List[State]: Time-optimal trajectory as list of states with positions and velocities
            
        Note:
            Handles path preprocessing, switch point detection, and trajectory sampling
        """
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
            # Initialize path
            self.debug.log_state("\nPreprocessing path points...")
            cleaned_path = self.preprocess_path(path_points)
            self.debug.log_state(f"Path reduced from {len(path_points)} to {len(cleaned_path)} points")
            
            self.debug.log_state("\nInitializing path parameterizer")
            self.path = PathParameterizer(cleaned_path)
            
            # Compute switch points
            self.debug.log_state("\nFinding switch points...")
            switch_points = self.find_switch_points()
            
            # Determine maximum valid path parameter
            max_valid_s = 0.0
            if switch_points:
                max_valid_s = switch_points[-1][0]
            else:
                _, trajectory = self.integrate_trajectory(0.0, 0.0, True, (0, 100.0), 1e-2)
                if trajectory.size > 0:
                    max_valid_s = np.max(trajectory[0])
                    
            self.debug.log_state(f"Maximum valid s value: {max_valid_s:.3f}")
            
            # Generate trajectory points
            self.debug.log_state("\nGenerating final trajectory...")
            trajectory_points = []
            
            # Create sampling points with higher density near switches
            s_values = []

            # Calculate path length and determine adaptive number of base points
            path_length = self.calculate_path_length()
            self.debug.log_state(f"Total path length: {path_length:.3f} meters")
            
            # Define adaptive number of base points based on path length
            # Use minimum of 20 points for very short paths
            min_points = 10
            max_points = 50
            num_base_points = min_points
            if path_length <= 50.0:
                num_base_points = min_points
            elif path_length > 50.0 and path_length <= 150.0:
                num_base_points = 20
            elif path_length > 150.0 and path_length <= 250:
                num_base_points = 30
            elif path_length > 250.0 and path_length < 350:
                num_base_points = 40
            else:
                num_base_points = max_points

            self.debug.log_state(f"Using {num_base_points} base points for trajectory")

            base_points = np.linspace(0, min(1.0, max_valid_s), num_base_points)
            
            for s in base_points:
                s_values.append(s)
                for s_switch, _, _ in switch_points:
                    if abs(s - s_switch) < 0.05:
                        s_values.extend([
                            s_switch
                        ])
            
            s_values = sorted(list(set(np.clip(s_values, 0, max_valid_s))))
            self.debug.log_state(f"Generated {len(s_values)} trajectory points")
            
            # Generate states for each sample point
            for s in s_values:
                try:
                    point = self.path.get_path_point(s)
                    s_dot = self.compute_optimal_velocity_at_s(s, switch_points)
                    
                   # Calculate s_ddot from acceleration limits
                    L, U = self.compute_limits(s, s_dot)
                    # Use appropriate acceleration based on current phase
                    current_phase = TrajectoryType.ACCELERATION
                    for s_switch, _, switch_type in switch_points:
                        if s < s_switch:
                            break
                        current_phase = switch_type
                    s_ddot = U if current_phase == TrajectoryType.ACCELERATION else L
                    
                    # Convert to joint velocities and accelerations using chain rule
                    # For velocities: θ_dot = dθ/ds * s_dot
                    omega_0 = point.dtheta_0 * s_dot
                    omega_1 = point.dtheta_1 * s_dot
                    
                    # For accelerations: θ_ddot = d²θ/ds² * s_dot² + dθ/ds * s_ddot
                    alpha_0 = point.ddtheta_0 * s_dot**2 + point.dtheta_0 * s_ddot
                    alpha_1 = point.ddtheta_1 * s_dot**2 + point.dtheta_1 * s_ddot
                    
                    # Apply velocity and acceleration limits
                    omega_0 = np.clip(omega_0, -self.robot.constants.MAX_VELOCITY, 
                                    self.robot.constants.MAX_VELOCITY)
                    omega_1 = np.clip(omega_1, -self.robot.constants.MAX_VELOCITY, 
                                    self.robot.constants.MAX_VELOCITY)
                    alpha_0 = np.clip(alpha_0, -self.robot.constants.MAX_ACCELERATION,
                                    self.robot.constants.MAX_ACCELERATION)
                    alpha_1 = np.clip(alpha_1, -self.robot.constants.MAX_ACCELERATION,
                                    self.robot.constants.MAX_ACCELERATION)
                    
                    new_state = State(
                        theta_0=point.theta_0,
                        theta_1=point.theta_1,
                        omega_0=omega_0,
                        omega_1=omega_1,
                        alpha_0=alpha_0,
                        alpha_1=alpha_1
                    )
                    trajectory_points.append(new_state)
                    self.debug.log_state(f"Generated Trajectory point for s={s:.3f}, s_dot={s_dot:.3f}, s_ddot={s_ddot:.3f}")
                    self.debug.log_state(f"State: {new_state}")
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
