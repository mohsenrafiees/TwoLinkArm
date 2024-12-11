from typing import List, Tuple, Optional
from enviroment.robot import Robot
from enviroment.world import World
from utils.state import State
from utils.debug_helper import DebugHelper


class SimpleAStarPlanner(GridBasedPlanner):
    """
    A simplified A* planner for 2-link robotic arm path planning.
    
    This planner implements a modified A* algorithm for finding optimal paths
    in the configuration space of a 2-link robotic arm, with collision checking
    and kinematic constraints.
    """
    def __init__(self, robot, world):
        """
        Initialize the A* planner with robot and world models.

        Args:
            robot: Robot object containing kinematic model and constraints
            world: World object containing obstacle information
        """
        super().__init__(robot, world)
        self.robot = robot
        self.world = world
        self.constants = robot.constants
        self.angle_resolution = 0.1  # Resolution for discretizing configuration space (rad)
        self.max_iterations = 10000  # Maximum iterations to prevent infinite loops
        self.collision_cache = {}    # Cache for collision check results
        
    def Plan(self, start: tuple, goal: tuple, robot, final_velocities=(0.0, 0.0)) -> list:
        """
        Plan a path from start to goal configuration using ARA* algorithm.

        Args:
            start (tuple): Start position in workspace coordinates (x, y)
            goal (tuple): Goal position in workspace coordinates (x, y)
            robot: Robot object for kinematics calculations
            final_velocities (tuple): Desired final joint velocities (omega0, omega1)

        Returns:
            float: Euclidean distance between states in joint space
        """
        d_theta0 = abs(state2.theta_0 - state1.theta_0)
        d_theta1 = abs(state2.theta_1 - state1.theta_1)
        return np.sqrt(d_theta0**2 + d_theta1**2)
        
    def is_goal_reached(self, current: State, goal: State) -> bool:
        """
        Check if current state is close enough to goal in workspace.

        Args:
            current (State): Current robot configuration
            goal (State): Goal robot configuration

        Returns:
            bool: True if end effector is within threshold of goal position
        """
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        
        distance = np.hypot(current_pos[0] - goal_pos[0],
                          current_pos[1] - goal_pos[1])
        return distance < 0.1  # 10cm threshold
        
    def reconstruct_path(self, came_from: dict, current: State) -> list:
        """
        Reconstruct path from start to current state using came_from map.

        Args:
            came_from (dict): Dictionary mapping states to their predecessors
            current (State): Final state in the path

        Returns:
            list: Sequence of States from start to current state
        """
        path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start-to-goal order:
            list: Sequence of States representing the planned path, empty if no path found

        # Convert workspace coordinates to joint angles
        start_angles = robot.inverse_kinematics(*start)
        goal_angles = robot.inverse_kinematics(*goal)
        
        start_state = State(start_angles[0], start_angles[1], 0.0, 0.0)
        goal_state = State(goal_angles[0], goal_angles[1], 
                         final_velocities[0], final_velocities[1])
        
        open_set = [(self.heuristic(start_state, goal_state), start_state)]
        closed_set = set()
        came_from = {start_state: None}
        g_score = {start_state: 0}
        
        # Main ARA* search loop with decreasing epsilon
        while epsilon >= self.epsilon_final:
            self.debug_helper.log_state(f"Starting search iteration with ε={epsilon:.3f}")
            solution = self._search_with_epsilon(start_state, goal_state, 
                                              epsilon, start_time)
            
            if solution:
                cost = self._compute_path_cost(solution)
                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost
                
                if time.time() - start_time > self.planning_timeout:
                    break
                
                epsilon = max(epsilon - self.delta_epsilon, self.epsilon_final)
            else:
                break
        
        return best_solution if best_solution else []

    def _search_with_epsilon(self, start_state: State, goal_state: State, 
                           epsilon: float, start_time: float) -> Optional[List[State]]:
        """
        Perform weighted A* search with specified epsilon value.

        Args:
            start_state (State): Initial robot configuration state
            goal_state (State): Target robot configuration state
            epsilon (float): Current weight for heuristic function
            start_time (float): Time when planning started

        Returns:
            Optional[List[State]]: Path from start to goal if found, None otherwise
        """
        open_set = []
        closed_set = set()
        g_scores = defaultdict(lambda: float('inf'))
        came_from = {}
        
        # Initialize search with start state
        g_scores[start_state] = 0
        f_score = g_scores[start_state] + epsilon * self.heuristic_computer.get_heuristic(
            start_state, goal_state)
        heapq.heappush(open_set, (f_score, id(start_state), start_state))
        
        while open_set and time.time() - start_time < self.planning_timeout:
            _, _, current = heapq.heappop(open_set)
            
            if self._is_goal(current, goal_state):
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Explore successor states
            for successor, cost in self._get_successors(current):
                if successor in closed_set:
                    continue
                
                tentative_g_score = g_scores[current] + cost
                
                if tentative_g_score < g_scores[successor]:
                    came_from[successor] = current
                    g_scores[successor] = tentative_g_score
                    f_score = tentative_g_score + epsilon * self.heuristic_computer.get_heuristic(
                        successor, goal_state)
                    heapq.heappush(open_set, (f_score, id(successor), successor))
        
        return None

    def _get_successors(self, state: State) -> List[Tuple[State, float]]:
        """
        Generate valid successor states using cached motion primitives.

        Args:
            state (State): Current robot configuration state

        Returns:
            List[Tuple[State, float]]: List of (successor_state, transition_cost) pairs
        """
        # Check cache first
        cache_key = state
        cached = self.successor_cache.get(cache_key)
        if cached is not None:
            return cached
            
        primitives = self.motion_primitives.primitives.get(round(state.theta_0, 3), [])
        candidates = []
        
        # Generate candidate successors from motion primitives
        for primitive in primitives:
            new_state = self._create_state(
                state.theta_0 + primitive['delta_angles'][0],
                state.theta_1 + primitive['delta_angles'][1],
                primitive['velocities'][0],
                primitive['velocities'][1]
            )
            candidates.append((new_state, primitive['cost_mult']))
        
        # Filter out states that would cause collisions
        valid_candidates = [
            (s, c) for s, c in candidates 
            if not self.collision_detector.check_collision(s, self.robot, self.world.obstacles)
        ]
        
        self.successor_cache.put(cache_key, valid_candidates)
        return valid_candidates

    def _is_goal(self, current: State, goal: State) -> bool:
        """
        Check if current state satisfies goal conditions.

        Args:
            current (State): Current robot configuration
            goal (State): Goal robot configuration

        Returns:
            bool: True if goal conditions are met, False otherwise
        """
        # Check end-effector position
        current_pos = self.robot.forward_kinematics(current.theta_0, current.theta_1)
        goal_pos = self.robot.forward_kinematics(goal.theta_0, goal.theta_1)
        
        pos_diff = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
        if pos_diff > self.pos_threshold:
            return False
        
        # Check joint velocities
        vel_diff_0 = abs(current.omega_0 - goal.omega_0)
        vel_diff_1 = abs(current.omega_1 - goal.omega_1)
        
        return (vel_diff_0 <= self.vel_threshold and 
                vel_diff_1 <= self.vel_threshold)
        
    def get_neighbors(self, state: State) -> list:
        """
        Generate neighboring states by discrete joint angle variations.

        Args:
            state (State): Current robot configuration

        Returns:
            list: List of valid neighbor States
        """
        neighbors = []
        angle_steps = [-self.angle_resolution, 0, self.angle_resolution]
        
        for d_theta0 in angle_steps:
            for d_theta1 in angle_steps:
                if d_theta0 == 0 and d_theta1 == 0:
                    continue
                    
                new_theta0 = state.theta_0 + d_theta0
                new_theta1 = state.theta_1 + d_theta1
                
                if self.within_joint_limits((new_theta0, new_theta1)):
                    neighbor = State(new_theta0, new_theta1, 0.0, 0.0)
                    neighbors.append(neighbor)
        
        return neighbors
        
    def within_joint_limits(self, angles: tuple) -> bool:
        """
        Check if joint angles are within robot's joint limits.

        Args:
            angles (tuple): Joint angles (theta0, theta1) to check

        Returns:
            bool: True if angles are within limits, False otherwise
        """
        theta_0, theta_1 = angles
        limits = self.constants.JOINT_LIMITS
        return (limits[0] <= theta_0 <= limits[1] and
                limits[0] <= theta_1 <= limits[1])
        
    def check_segment_collision(self, segment: tuple, obstacle) -> bool:
        """
        Check collision between a line segment and an obstacle.

        Args:
            segment (tuple): Line segment endpoints ((x1,y1), (x2,y2))
            obstacle: Obstacle object with shape and size properties

        Returns:
            bool: True if collision detected, False otherwise
        """
        if obstacle.shape == 'circle':
            return self.check_circle_collision(segment, obstacle)
        elif obstacle.shape == 'rectangle':
            return self.check_rectangle_collision(segment, obstacle)
        return False
        
    def check_circle_collision(self, segment: tuple, obstacle) -> bool:
        """
        Check collision between line segment and circular obstacle.

        Args:
            segment (tuple): Line segment endpoints ((x1,y1), (x2,y2))
            obstacle: Circle obstacle with position and size properties

        Returns:
            bool: True if collision detected, False otherwise
        """
        p1, p2 = segment
        circle_center = obstacle.position
        circle_radius = obstacle.size
        
        # Compute closest point on line segment to circle center
        line_vec = (p2[0] - p1[0], p2[1] - p1[1])
        line_length = np.hypot(line_vec[0], line_vec[1])
        line_unit_vec = (line_vec[0] / line_length, line_vec[1] / line_length)
        
        circle_vec = (circle_center[0] - p1[0], circle_center[1] - p1[1])
        projection = circle_vec[0] * line_unit_vec[0] + circle_vec[1] * line_unit_vec[1]
        
        if projection < 0:
            closest_point = p1
        elif projection > line_length:
            closest_point = p2
        else:
            closest_point = (p1[0] + line_unit_vec[0] * projection,
                           p1[1] + line_unit_vec[1] * projection)
            
        distance = np.hypot(circle_center[0] - closest_point[0],
                          circle_center[1] - closest_point[1])
                          
        return distance <= circle_radius
        
    def check_rectangle_collision(self, segment: tuple, obstacle) -> bool:
        """
        Check collision between line segment and rectangular obstacle.

        Args:
            segment (tuple): Line segment endpoints ((x1,y1), (x2,y2))
            obstacle: Rectangle obstacle with position and size properties

        Returns:
            bool: True if collision detected, False otherwise
        """
        p1, p2 = segment
        rect_x, rect_y = obstacle.position
        rect_w, rect_h = obstacle.size
        
        # Convert to axis-aligned bounding box
        rect_left = rect_x - rect_w/2
        rect_right = rect_x + rect_w/2
        rect_bottom = rect_y - rect_h/2
        rect_top = rect_y + rect_h/2
        
        # Check endpoints inside rectangle
        if (rect_left <= p1[0] <= rect_right and rect_bottom <= p1[1] <= rect_top) or \
           (rect_left <= p2[0] <= rect_right and rect_bottom <= p2[1] <= rect_top):
            return True
            
        # Check intersection with rectangle edges
        edges = [
            ((rect_left, rect_bottom), (rect_right, rect_bottom)),
            ((rect_right, rect_bottom), (rect_right, rect_top)),
            ((rect_right, rect_top), (rect_left, rect_top)),
            ((rect_left, rect_top), (rect_left, rect_bottom))
        ]
        
        return any(self.line_segments_intersect(segment, edge) for edge in edges)
        
    def line_segments_intersect(self, seg1: tuple, seg2: tuple) -> bool:
        """
        Check if two line segments intersect.

        Args:
            seg1 (tuple): First line segment endpoints ((x1,y1), (x2,y2))
            seg2 (tuple): Second line segment endpoints ((x3,y3), (x4,y4))

        Returns:
            bool: True if segments intersect, False otherwise
        """
        p1, p2 = seg1
        p3, p4 = seg2
        
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        d2 = (p4[0] - p3[0], p4[1] - p3[1])
        
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(cross) < 1e-8:  # Lines are parallel
            return False
            
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
        
        return 0 <= t <= 1 and 0 <= u <= 1
        
    def heuristic(self, state: State, goal: State) -> float:
        """
        Compute heuristic distance between states.

        Args:
            state (State): Current robot configuration
            goal (State): Goal robot configuration

        Returns:
            float: Estimated cost to reach goal from current state
        """
        d_theta0 = abs(goal.theta_0 - state.theta_0)
        d_theta1 = abs(goal.theta_1 - state.theta_1)
        return np.sqrt(d_theta0**2 + d_theta1**2)
        
    def distance(self, state1: State, state2: State) -> float:
        """
        Compute Euclidean distance between states in joint space.

        Args:
            state1 (State): First robot configuration
            state2 (State): Second robot configuration

        Returns
        """
        d_theta0 = abs(state2.theta_0 - state1.theta_0)
        d_theta1 = abs(state2.theta_1 - state1.theta_1)
        return np.sqrt(d_theta0**2 + d_theta1**2)