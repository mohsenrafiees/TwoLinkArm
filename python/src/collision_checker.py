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
