import numpy as np
from typing import Tuple

class CollisionChecker:
    """
    Provides collision detection methods.
    """

    @staticmethod
    def line_circle_collision(p1: Tuple[float, float], p2: Tuple[float, float],
                              circle_center: Tuple[float, float], radius: float) -> bool:
        """
        Checks for collision between a line segment and a circle.

        Args:
            p1: Start point of the line segment (x1, y1).
            p2: End point of the line segment (x2, y2).
            circle_center: Center of the circle (cx, cy).
            radius: Radius of the circle.

        Returns:
            True if there is a collision, False otherwise.
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = circle_center

        # Shift coordinates so that circle is at the origin
        x1 -= cx
        y1 -= cy
        x2 -= cx
        y2 -= cy

        dx = x2 - x1
        dy = y2 - y1

        a = dx * dx + dy * dy
        b = 2 * (x1 * dx + y1 * dy)
        c = x1 * x1 + y1 * y1 - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return False
        else:
            discriminant_sqrt = np.sqrt(discriminant)
            t1 = (-b - discriminant_sqrt) / (2 * a)
            t2 = (-b + discriminant_sqrt) / (2 * a)

            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True

        return False


    @staticmethod
    def line_rectangle_collision(p1: Tuple[float, float], p2: Tuple[float, float],
                                 rect_pos: Tuple[float, float], rect_size: Tuple[float, float]) -> bool:
        """
        Checks for collision between a line segment and a rectangle.

        Args:
            p1: Start point of the line segment (x1, y1).
            p2: End point of the line segment (x2, y2).
            rect_pos: Position of the rectangle's bottom-left corner (rx, ry).
            rect_size: Size of the rectangle (width, height).

        Returns:
            True if there is a collision, False otherwise.
        """
        rx, ry = rect_pos
        rw, rh = rect_size

        left = rx
        right = rx + rw
        bottom = ry
        top = ry + rh

        # Edges of the rectangle
        edges = [
            ((left, bottom), (right, bottom)),  # Bottom edge
            ((right, bottom), (right, top)),    # Right edge
            ((right, top), (left, top)),        # Top edge
            ((left, top), (left, bottom)),      # Left edge
        ]

        for edge_start, edge_end in edges:
            if CollisionChecker.line_line_collision(p1, p2, edge_start, edge_end):
                return True

        return False

    @staticmethod
    def line_line_collision(p1: Tuple[float, float], p2: Tuple[float, float],
                            q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        """
        Checks for collision between two line segments.

        Args:
            p1, p2: Endpoints of the first line segment.
            q1, q2: Endpoints of the second line segment.

        Returns:
            True if the line segments intersect, False otherwise.
        """
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    @staticmethod
    def distance_line_to_circle(p1: Tuple[float, float], p2: Tuple[float, float],
                                circle_center: Tuple[float, float], radius: float) -> float:
        """
        Computes the minimum distance between a line segment and a circle.
        If the line segment intersects the circle, returns zero.

        Args:
            p1: Start point of the line segment (x1, y1).
            p2: End point of the line segment (x2, y2).
            circle_center: Center of the circle (cx, cy).
            radius: Radius of the circle.

        Returns:
            Minimum distance between the line segment and the circle's perimeter.
            Returns zero if the line segment intersects the circle.
        """
        # First, check if there is a collision
        if CollisionChecker.line_circle_collision(p1, p2, circle_center, radius):
            return 0.0

        # Compute the closest distance
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = circle_center

        # Line segment vector
        dx = x2 - x1
        dy = y2 - y1

        # Early exit if the segment is a point
        if dx == 0 and dy == 0:
            return max(np.hypot(x1 - cx, y1 - cy) - radius, 0.0)

        # Parameterize the line
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))

        # Closest point on the segment to the circle center
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance from the closest point to the circle center
        distance_to_center = np.hypot(closest_x - cx, closest_y - cy)

        # Subtract the radius to get distance to the perimeter
        distance = max(distance_to_center - radius, 0.0)

        return distance

    @staticmethod
    def distance_line_to_rectangle(p1: Tuple[float, float], p2: Tuple[float, float],
                                   rect_pos: Tuple[float, float], rect_size: Tuple[float, float]) -> float:
        """
        Computes the minimum distance between a line segment and a rectangle.
        If the line segment intersects the rectangle, returns zero.

        Args:
            p1: Start point of the line segment (x1, y1).
            p2: End point of the line segment (x2, y2).
            rect_pos: Position of the rectangle's bottom-left corner (rx, ry).
            rect_size: Size of the rectangle (width, height).

        Returns:
            Minimum distance between the line segment and the rectangle.
            Returns zero if the line segment intersects the rectangle.
        """
        # First, check if there is a collision
        if CollisionChecker.line_rectangle_collision(p1, p2, rect_pos, rect_size):
            return 0.0

        rx, ry = rect_pos
        rw, rh = rect_size

        # Rectangle corners
        corners = [
            (rx, ry),             # Bottom-left
            (rx + rw, ry),        # Bottom-right
            (rx + rw, ry + rh),   # Top-right
            (rx, ry + rh),        # Top-left
        ]

        # Edges of the rectangle
        edges = [
            (corners[0], corners[1]),  # Bottom edge
            (corners[1], corners[2]),  # Right edge
            (corners[2], corners[3]),  # Top edge
            (corners[3], corners[0]),  # Left edge
        ]

        min_distance = float('inf')

        # Compute distance from the line segment to each edge
        for edge_start, edge_end in edges:
            distance = CollisionChecker.distance_between_segments(p1, p2, edge_start, edge_end)
            if distance < min_distance:
                min_distance = distance

        # Also consider distances to rectangle corners
        for corner in corners:
            distance = CollisionChecker.distance_point_to_segment(corner, p1, p2)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    @staticmethod
    def distance_between_segments(p1: Tuple[float, float], p2: Tuple[float, float],
                                  q1: Tuple[float, float], q2: Tuple[float, float]) -> float:
        """
        Computes the minimum distance between two line segments.

        Args:
            p1, p2: Endpoints of the first line segment.
            q1, q2: Endpoints of the second line segment.

        Returns:
            The minimum distance between the two line segments.
        """
        # If the segments intersect, the distance is zero
        if CollisionChecker.line_line_collision(p1, p2, q1, q2):
            return 0.0

        # Compute distances between each endpoint and the other segment
        distances = [
            CollisionChecker.distance_point_to_segment(p1, q1, q2),
            CollisionChecker.distance_point_to_segment(p2, q1, q2),
            CollisionChecker.distance_point_to_segment(q1, p1, p2),
            CollisionChecker.distance_point_to_segment(q2, p1, p2),
        ]
        return min(distances)

    @staticmethod
    def distance_point_to_segment(point: Tuple[float, float],
                                  seg_start: Tuple[float, float],
                                  seg_end: Tuple[float, float]) -> float:
        """
        Computes the minimum distance between a point and a line segment.

        Args:
            point: The point (x, y).
            seg_start: Start point of the segment (x1, y1).
            seg_end: End point of the segment (x2, y2).

        Returns:
            The minimum distance between the point and the line segment.
        """
        x, y = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        # Line segment vector
        dx = x2 - x1
        dy = y2 - y1

        # Early exit if the segment is a point
        if dx == 0 and dy == 0:
            return np.hypot(x - x1, y - y1)

        # Parameter t for the projection onto the segment
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))

        # Closest point on the segment to the point
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance from the point to the closest point on the segment
        distance = np.hypot(x - closest_x, y - closest_y)
        return distance
