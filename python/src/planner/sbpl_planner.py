from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import math
import time
import heapq
from collections import defaultdict
from planner.grid_planner import GridBasedPlanner
from enviroment.robot import Robot
from enviroment.world import World
from utils.state import State
from utils.debug_helper import DebugHelper
from trajectory.generator import TimeOptimalTrajectoryGenerator
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra


class SBPLPlanner(GridBasedPlanner):
    """
    A Search-Based Planning Library (SBPL) implementation using ARA* for robotic arm planning.
    https://ieeexplore.ieee.org/abstract/document/5509685?casa_token=GYM8VXnRYDoAAAAA:Oyu6BECo5gdCiIl2HW9MbdakJXRmyYPkQgD3uInck-I-2952d-OsYaHaJDKUUN9_cgoL2uILEA
    
    This planner implements Anytime Repairing A* (ARA*) with optimizations including:
    - Adaptive resolution based on distance to obstacles
    - Efficient motion primitive generation
    - Precomputed heuristics using Dijkstra's algorithm
    - State and successor caching for improved performance
    
    Attributes:
        discrete_env (DiscreteEnvironment): Handles discretization of continuous space
        motion_primitives (MotionPrimitives): Generates and stores motion primitives
        heuristic_computer (HeuristicComputer): Computes and caches heuristic values
        epsilon (float): Current suboptimality bound
        state_cache (StateCache): Caches computed states
        successor_cache (StateCache): Caches successor states
    """

    def __init__(self, robot: Robot, world: World, config: Dict):
        """
        Initialize the SBPL planner with robot model, world representation, and configuration.

        Args:
            robot (Robot): Robot model containing kinematics and physical parameters
            world (World): World model containing obstacles and boundaries
            config (Dict): Configuration parameters including:
                - debug (bool): Enable debug output
                - initial_epsilon (float): Initial suboptimality bound
                - final_epsilon (float): Final desired suboptimality
                - delta_epsilon (float): Epsilon decrease per iteration
                - max_iterations (int): Maximum planning iterations
                - planning_timeout (float): Maximum planning time in seconds
                - state_cache_size (int): Maximum states to cache
                - successor_cache_size (int): Maximum successors to cache
        """
        super().__init__(robot, world)
        self.debug_helper = DebugHelper(config.get('debug', True))
        
        # Initialize discrete environment and primitives
        self.discrete_env = DiscreteEnvironment(
            resolution=robot.constants.THETA_0_RESOLUTION,
            bounds=robot.constants.JOINT_LIMITS
        )
        self.debug_helper.log_state(f"Initialized DiscreteEnvironment with resolution={robot.constants.THETA_0_RESOLUTION}")

        self.num_angle_primitives = 36
        self.num_velocity_primitives = 16
        self.motion_primitives = MotionPrimitives(robot.constants, self.num_angle_primitives, 
                                                self.num_velocity_primitives)
        self.heuristic_computer = HeuristicComputer(self.discrete_env)
        
        # Store configuration
        self.robot = robot
        self.world = world
        self.config = config
        
        # Planning parameters
        self.pos_threshold = 1.0
        self.vel_threshold = 1.0
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.min_cost = 0.3
        self.epsilon_init = config.get('initial_epsilon', 10.0)
        self.epsilon_final = config.get('final_epsilon', 1.0)
        self.delta_epsilon = config.get('delta_epsilon', 3.0)
        self.epsilon = self.epsilon_init
        self.max_iterations = config.get('max_iterations', 100000)
        self.planning_timeout = config.get('planning_timeout', 10.0)
        self.dist_to_goal = float('inf')
        self.best_cost = float('inf')

        # Initialize caches
        self.state_cache = StateCache(max_size=config.get('state_cache_size', 10000))
        self.successor_cache = StateCache(max_size=config.get('successor_cache_size', 5000))

        # Generate motion primitives
        self.debug_helper.log_state("Generating motion primitives...")
        self.motion_primitives.generate_primitives()
        self.debug_helper.log_state(f"Generated {len(self.motion_primitives.primitives)} sets of primitives")
        self.debug_helper.log_state(f"Angle resolution: {self.discrete_env.resolution}")

    def Plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
            final_velocities: Tuple[float, float] = (0.0, 0.0)) -> List[State]:
        """
        Plan a path using ARA* with iteratively decreasing epsilon values.
        
        Implements Anytime Repairing A* to find a series of solutions with 
        decreasing suboptimality bounds.

        Args:
            start (Tuple[float, float]): Start position in workspace (x, y)
            goal (Tuple[float, float]): Goal position in workspace (x, y)
            final_velocities (Tuple[float, float]): Desired final velocities (default: (0.0, 0.0))

        Returns:
            List[State]: Planned trajectory or empty list if planning fails

        Raises:
            ValueError: If start or goal positions contain NaN values
        """
        # Input validation
        if any(map(math.isnan, start + goal + final_velocities)):
            self.debug_helper.log_state("Error: NaN values in input parameters")
            return []

        # Convert workspace positions to joint space
        goal_theta = self.robot.inverse_kinematics(*goal)
        start_theta = self.robot.inverse_kinematics(*start)
        start_state = State(start_theta[0], start_theta[1], 0.0, 0.0)
        goal_state = State(goal_theta[0], goal_theta[1], 0.0, 0.0)

        self.debug_helper.log_state(
            f"Planning from {start} to {goal}")
        self.debug_helper.log_state(f"Joint limits: {self.robot.constants.JOINT_LIMITS}")

        if start_state is None or goal_state is None:
            self.debug_helper.log_state("Error: Invalid start/goal states")
            return []

        # Initialize search
        self.debug_helper.log_state("Precomputing heuristics...")
        self.heuristic_computer.precompute_heuristic_map(goal_state)

        best_solution = None
        best_cost = float('inf')
        start_time = time.time()
        
        # Define epsilon schedule
        epsilons = [self.epsilon_init, 
                   self.epsilon_init - self.delta_epsilon,
                   self.epsilon_init - 2 * self.delta_epsilon,
                   3.0, 2.5, 2.0, 1.5, 1.15, 1.05, 
                   self.epsilon_final]

        # ARA* main loop
        for epsilon in epsilons:
            self.debug_helper.log_state(f"Search iteration with ε={epsilon:.3f}")
            self.epsilon = epsilon
            solution = self._search_with_epsilon(start_state, goal_state, 
                                              epsilon, start_time)

            if solution:
                cost = self.dist_to_goal
                self.debug_helper.log_state(f"Solution found, cost={cost:.3f}")

                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost
                    self.best_cost = best_cost
                    self.debug_helper.log_state(f"New best solution, cost={best_cost:.3f}")

                if best_cost < self.min_cost or \
                   time.time() - start_time > self.planning_timeout:
                    break
            else:
                self.debug_helper.log_state(f"No solution with ε={epsilon:.3f}")

        return best_solution if best_solution else []

    def _search_with_epsilon(self, start_state: State, goal_state: State, 
                           epsilon: float, start_time: float) -> Optional[List[State]]:
        """
        Perform weighted A* search with current epsilon value.

        Implements the core search routine of ARA* for a specific epsilon value,
        using weighted heuristics and maintaining admissibility.

        Args:
            start_state (State): Initial robot state
            goal_state (State): Target robot state
            epsilon (float): Current suboptimality bound
            start_time (float): Time when planning started

        Returns:
            Optional[List[State]]: Found path or None if no path exists
        """
        open_set = []
        closed_set = set()
        g_scores = defaultdict(lambda: float('inf'))
        came_from = {}
        current = None
        iterations = 0
        
        # Initialize search
        g_scores[start_state] = 0
        f_score = epsilon * self.heuristic_computer.get_heuristic(start_state, goal_state)
        heapq.heappush(open_set, (f_score, id(start_state), start_state))
        
        while iterations < self.max_iterations /(min(4.0, epsilon)) and open_set and time.time() - start_time < self.planning_timeout:
            iterations += 1
            if iterations % 4000 == 0:
                self.debug_helper.log_state(f"Iteration {iterations}, open={len(open_set)}")
            
            _, _, current = heapq.heappop(open_set)
            
            if self._is_goal(current, goal_state):
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Expand neighbors
            successors = self._get_successors(current)
            for successor, cost in successors:
                if successor in closed_set:
                    continue
                
                tentative_g_score = g_scores[current] + cost
                if tentative_g_score < g_scores[successor]:
                    came_from[successor] = current
                    g_scores[successor] = tentative_g_score
                    f_score = tentative_g_score + epsilon * \
                             self.heuristic_computer.get_heuristic(successor, goal_state)
                    heapq.heappush(open_set, (f_score, id(successor), successor))
        
        
        if self.dist_to_goal < self.best_cost:
            return self._reconstruct_path(came_from, current)
        else:
            return None

    def _get_successors(self, state: State) -> List[Tuple[State, float]]:
        """
        Generate successor states using motion primitives.
        
        Computes dynamically feasible successor states using precomputed motion
        primitives and performs collision checking.

        Args:
            state (State): Current robot state

        Returns:
            List[Tuple[State, float]]: List of (successor_state, transition_cost) pairs
        """
        if state is None or any(map(math.isnan, [state.theta_0, state.theta_1, 
                                                state.omega_0, state.omega_1])):
            return []

        # Check cache
        cache_key = state
        cached = self.successor_cache.get(cache_key)
        if cached is not None:
            return cached

        # Calculate primitive angle range
        min_primitive = self.robot.constants.JOINT_LIMITS[0] 
        max_primitive = self.robot.constants.JOINT_LIMITS[1] 
        primitive_range = np.linspace(min_primitive, max_primitive, 
                                    self.num_angle_primitives)

        # Find nearest primitive
        closest_primitive = min(primitive_range, 
                              key=lambda x: abs(x - state.theta_0))
        rounded_primitive = round(closest_primitive, 3)
        
        primitives = self.motion_primitives.primitives.get(rounded_primitive, [])
        
        # Generate candidates
        candidates = []
        for primitive in primitives:
            new_state = self._create_state(
                state.theta_0 + primitive['delta_angles'][0],
                state.theta_1 + primitive['delta_angles'][1],
                primitive['velocities'][0],
                primitive['velocities'][1]
            )
            if new_state is not None:
                candidates.append((new_state, primitive['cost_mult']))

        # Filter valid successors
        valid_candidates = [
            (s, c) for s, c in candidates 
            if not self.is_collision(s)
        ]
        
        self.successor_cache.put(cache_key, valid_candidates)
        return valid_candidates

    def _is_goal(self, current: State, goal: State) -> bool:
        """
        Check if current state satisfies goal conditions.
        
        Uses adaptive thresholds based on current epsilon value for both
        position and velocity constraints.

        Args:
            current (State): Current robot state
            goal (State): Goal robot state

        Returns:
            bool: True if goal conditions are met
        """
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        
        pos_diff = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
        self.dist_to_goal = pos_diff
        
        # Compute adaptive thresholds
        if self.epsilon < 2.0:
            pos_threshold_max = 25.0 * self.pos_threshold
            pos_threshold_min = 2.0 * self.pos_threshold
            ep_max, ep_min = 1.5, 1.0
        else:
            pos_threshold_max = 50.0 * self.pos_threshold
            pos_threshold_min = 10.0 * self.pos_threshold
            ep_max, ep_min = 10.0, 2.0
            
        position_threshold = pos_threshold_min + \
                           (pos_threshold_max - pos_threshold_min) * \
                           (self.epsilon - ep_min) / (ep_max - ep_min)
      
        return pos_diff <= position_threshold

    def _reconstruct_path(self, came_from: Dict[State, State], current: State) -> List[State]:
        """
        Reconstruct a valid path from the search results using the provided 'came_from' mapping.
        
        Args:
            came_from (Dict[State, State]): Mapping from each visited state to the state from which 
                                             it was reached.
            current (State): The end state from which to start reconstructing the path.
        
        Returns:
            List[State]: A list of validated states representing the reconstructed path from start 
                         to end.
        """
        if current is None:
            self.debug_helper.log_state("Warning: Null current state in path reconstruction")
            return []
            
        path = [current]
        while current in came_from:
            current = came_from[current]
            if current is None:
                self.debug_helper.log_state("Warning: Null state encountered in path")
                break

            # Validate for NaN values in joint angles or velocities
            if any(map(math.isnan, [current.theta_0, current.theta_1, current.omega_0, current.omega_1])):
                self.debug_helper.log_state("Warning: NaN values detected in path")
                break
                
            # Log each newly added state for debugging
            # self.debug_helper.log_state(f"New point on path={current}")
            path.append(current)
        
        # Filter out invalid or None states from the final reconstructed path
        valid_path = [
            state for state in reversed(path)
            if state is not None and all(map(math.isfinite, [state.theta_0, state.theta_1, state.omega_0, state.omega_1]))
        ]
        
        # If some states were removed due to invalidity, log a warning
        if len(valid_path) < len(path):
            self.debug_helper.log_state(f"Warning: Removed {len(path) - len(valid_path)} invalid states from path")
        
        return valid_path
    
    def _create_state(self, theta0: float, theta1: float, 
                     omega0: float, omega1: float) -> Optional[State]:
        """
        Create a new state with bounds checking and validation.

        Args:
            theta0 (float): First joint angle
            theta1 (float): Second joint angle
            omega0 (float): First joint velocity
            omega1 (float): Second joint velocity

        Returns:
            Optional[State]: New valid state or None if invalid
        """
        # Handle NaN values
        if any(map(math.isnan, [theta0, theta1, omega0, omega1])):
            theta0 = 0.0 if math.isnan(theta0) else theta0
            theta1 = 0.0 if math.isnan(theta1) else theta1
            omega0 = 0.0 if math.isnan(omega0) else omega0
            omega1 = 0.0 if math.isnan(omega1) else omega1

        # Validate finite values
        if not all(map(math.isfinite, [theta0, theta1, omega0, omega1])):
            return None

        # Apply joint limits
        theta0 = np.clip(theta0, self.robot.constants.JOINT_LIMITS[0], 
                        self.robot.constants.JOINT_LIMITS[1])
        theta1 = np.clip(theta1, self.robot.constants.JOINT_LIMITS[0], 
                        self.robot.constants.JOINT_LIMITS[1])
        
        # Apply velocity limits
        omega0 = np.clip(omega0, -self.robot.constants.MAX_VELOCITY, 
                        self.robot.constants.MAX_VELOCITY)
        omega1 = np.clip(omega1, -self.robot.constants.MAX_VELOCITY, 
                        self.robot.constants.MAX_VELOCITY)
        
        return State(theta0, theta1, omega0, omega1)


class MotionPrimitives:
    """
    Generates and manages motion primitives for state expansion.
    
    Implements efficient generation of motion primitives using vectorized operations
    and maintains a cache of generated primitives.
    
    Attributes:
        constants: Robot kinematic constants
        primitives (Dict): Cached motion primitives
        primitive_cache (StateCache): Cache for computed primitives
        num_angle_primitives (int): Number of angle discretizations
        num_velocity_primitives (int): Number of velocity discretizations
    """
    
    def __init__(self, robot_constants, num_angle_primitives: int, num_velocity_primitive: int):
        """
        Initialize motion primitive generator.

        Args:
            robot_constants: Robot kinematic constants
            num_angle_primitives (int): Number of angle discretizations
            num_velocity_primitive (int): Number of velocity discretizations
        """
        self.constants = robot_constants
        self.primitives = {}
        self.primitive_cache = StateCache()
        self.num_angle_primitives = num_angle_primitives
        self.num_velocity_primitives = num_velocity_primitive
        
    def generate_primitives(self):
        """
        Generate motion primitives using vectorized operations.
        
        Creates a set of motion primitives for each discretized angle,
        using efficient numpy operations for computation.
        """
        angles = np.linspace(self.constants.JOINT_LIMITS[0], 
                           self.constants.JOINT_LIMITS[1],
                           self.num_angle_primitives)
        velocities = np.linspace(-self.constants.MAX_VELOCITY / 2,
                               self.constants.MAX_VELOCITY / 2,
                               self.num_velocity_primitives)
        
        for angle in angles:
            primitives = self._generate_angle_primitives(angle, velocities)
            self.primitives[round(angle, 3)] = primitives
    
    def _generate_angle_primitives(self, angle: float, 
                                 velocities: np.ndarray) -> List[Dict]:
        """
        Generate primitives for a specific angle using vectorized operations.

        Args:
            angle (float): Base angle for primitive generation
            velocities (np.ndarray): Array of possible velocities

        Returns:
            List[Dict]: List of primitive dictionaries containing velocities,
                       delta angles, and cost multipliers
        """
        primitive_list = []
        v0, v1 = np.meshgrid(velocities, velocities)
        
        # Compute position changes
        delta_theta0 = v0 * self.constants.DT
        delta_theta1 = v1 * self.constants.DT
        
        # Check feasibility
        mask = self._check_feasibility_vectorized(angle, delta_theta0, 
                                                delta_theta1, v0, v1)
        
        # Create primitives for valid combinations
        valid_indices = np.argwhere(mask)
        for idx in valid_indices:
            i, j = idx
            primitive_list.append({
                'velocities': (v0[i,j], v1[i,j]),
                'delta_angles': (delta_theta0[i,j], delta_theta1[i,j]),
                'cost_mult': self._compute_cost(v0[i,j], v1[i,j])
            })
            
        return primitive_list
    
    def _check_feasibility_vectorized(self, angle: float, delta_theta0: np.ndarray,
                                    delta_theta1: np.ndarray, v0: np.ndarray, 
                                    v1: np.ndarray) -> np.ndarray:
        """
        Vectorized feasibility checking for motion primitives.

        Args:
            angle (float): Base angle
            delta_theta0 (np.ndarray): Change in first joint angle
            delta_theta1 (np.ndarray): Change in second joint angle
            v0 (np.ndarray): First joint velocities
            v1 (np.ndarray): Second joint velocities

        Returns:
            np.ndarray: Boolean mask of feasible primitives
        """
        new_theta0 = angle + delta_theta0
        new_theta1 = angle + delta_theta1
        
        # Joint limits check
        limits_check = (new_theta0 >= self.constants.JOINT_LIMITS[0]) & \
                      (new_theta0 <= self.constants.JOINT_LIMITS[1]) & \
                      (new_theta1 >= self.constants.JOINT_LIMITS[0]) & \
                      (new_theta1 <= self.constants.JOINT_LIMITS[1])
        
        # Velocity limits check
        velocity_check = (np.abs(v0) <= self.constants.MAX_VELOCITY) & \
                        (np.abs(v1) <= self.constants.MAX_VELOCITY)
        
        return limits_check & velocity_check
    
    def _compute_cost(self, v0: float, v1: float) -> float:
        """
        Compute cost multiplier for a primitive.

        Args:
            v0 (float): First joint velocity
            v1 (float): Second joint velocity

        Returns:
            float: Cost multiplier based on velocity magnitudes
        """
        v_norm = (v0**2 + v1**2) / (2 * self.constants.MAX_VELOCITY**2)
        return 1.0 + v_norm

class DiscreteEnvironment:
    """
    Handles discretization of continuous state space.
    
    Provides efficient conversion between continuous and discrete coordinates
    with precomputed mappings.
    """
    
    def __init__(self, resolution: float, bounds: Tuple[float, float]):
        """
        Initialize discrete environment.

        Args:
            resolution (float): Grid cell size
            bounds (Tuple[float, float]): Minimum and maximum coordinate values
        """
        self.resolution = resolution
        self.bounds = bounds
        self.cell_count = np.ceil((bounds[1] - bounds[0]) / resolution).astype(int)
        self.coordinate_to_cell = np.zeros(self.cell_count + 1)
        self._precompute_coordinate_mappings()
    
    def _precompute_coordinate_mappings(self):
        """Precompute continuous to discrete coordinate mappings."""
        for i in range(self.cell_count + 1):
            self.coordinate_to_cell[i] = self.bounds[0] + i * self.resolution
    
    def continuous_to_discrete(self, coord: float) -> int:
        """
        Convert continuous coordinate to discrete cell index.

        Args:
            coord (float): Continuous coordinate value

        Returns:
            int: Discrete cell index
        """
        if coord <= self.bounds[0]:
            return 0
        if coord >= self.bounds[1]:
            return self.cell_count - 1
        return int((coord - self.bounds[0]) / self.resolution)
    
    def discrete_to_continuous(self, cell: int) -> float:
        """
        Convert discrete cell index to continuous coordinate.

        Args:
            cell (int): Discrete cell index

        Returns:
            float: Continuous coordinate value
        """
        return self.coordinate_to_cell[cell]

class HeuristicComputer:
    """
    Computes and caches heuristic values for the planning search.
    
    Uses Dijkstra's algorithm on a discretized workspace to precompute
    admissible heuristic values for A* search.

    Attributes:
        discrete_env (DiscreteEnvironment): Environment discretization
        heuristic_cache (StateCache): Cache for computed heuristic values
        distance_maps (Dict): Precomputed distance maps
    """
    
    def __init__(self, discrete_env: DiscreteEnvironment):
        """
        Initialize heuristic computer.

        Args:
            discrete_env (DiscreteEnvironment): Discretized environment
        """
        self.discrete_env = discrete_env
        self.heuristic_cache = StateCache()
        self.distance_maps = {}
        
    def precompute_heuristic_map(self, goal_state: State):
        """
        Precompute heuristic values using Dijkstra's algorithm.

        Args:
            goal_state (State): Goal state for heuristic computation
        """
        # Convert goal state to discrete coordinates
        goal_cell = (self.discrete_env.continuous_to_discrete(goal_state.theta_0),
                    self.discrete_env.continuous_to_discrete(goal_state.theta_1))
        
        # Create sparse adjacency matrix
        data, rows, cols = [], [], []
        N = self.discrete_env.cell_count**2
        
        # Build connectivity graph
        for i in range(self.discrete_env.cell_count):
            for j in range(self.discrete_env.cell_count):
                idx = i * self.discrete_env.cell_count + j
                # Include diagonal connections
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1), 
                             (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.discrete_env.cell_count and \
                       0 <= nj < self.discrete_env.cell_count:
                        nidx = ni * self.discrete_env.cell_count + nj
                        cost = np.sqrt(di*di + dj*dj)
                        data.append(cost)
                        rows.append(idx)
                        cols.append(nidx)
        
        # Compute distances using Dijkstra's algorithm
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        start_idx = goal_cell[0] * self.discrete_env.cell_count + goal_cell[1]
        distances = dijkstra(matrix, indices=start_idx)
        
        # Store reshaped distance map
        self.distance_maps[goal_state] = distances.reshape(
            self.discrete_env.cell_count,
            self.discrete_env.cell_count
        )
    
    def get_heuristic(self, state: State, goal_state: State) -> float:
        """
        Get heuristic value for a state pair.

        Args:
            state (State): Current state
            goal_state (State): Goal state

        Returns:
            float: Heuristic estimate of cost to goal
        """
        cache_key = (state, goal_state)
        cached = self.heuristic_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Get value from precomputed map
        state_cell = (self.discrete_env.continuous_to_discrete(state.theta_0),
                     self.discrete_env.continuous_to_discrete(state.theta_1))
        
        value = self.distance_maps[goal_state][state_cell[0], state_cell[1]]
        self.heuristic_cache.put(cache_key, value)
        return value


class StateCache:
    """
    Cache implementation for storing computed states and values.
    
    Implements a simple LRU-style cache with size limiting and statistics tracking.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize state cache.

        Args:
            max_size (int): Maximum number of entries to store
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key, default = None):
        """
        Retrieve value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return default
    
    def put(self, key, value):
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to store
        """
        if len(self.cache) >= self.max_size:
            # Remove oldest 25% of entries
            items = list(self.cache.items())
            self.cache = dict(items[len(items)//4:])
        self.cache[key] = value


