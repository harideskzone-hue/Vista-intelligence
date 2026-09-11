"""
Geometry Primitives & Spatial Utilities for Boundary Crossing & Zone Detection.
"""

from dataclasses import dataclass
import math
import cv2
import numpy as np
from typing import Tuple, List, Optional


@dataclass
class Point:
    x: float
    y: float

    @property
    def as_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class Line:
    p1: Point
    p2: Point


@dataclass
class NormalizedLine:
    """Normalized line coordinates [0.0, 1.0] independent of screen/camera resolution."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixel_line(self, width: int, height: int) -> Line:
        """Projects normalized [0.0, 1.0] coordinates to actual pixel dimensions."""
        px1 = Point(self.x1 * width, self.y1 * height)
        px2 = Point(self.x2 * width, self.y2 * height)
        return Line(px1, px2)


def side_of_point(line: Line, point: Point) -> int:
    """
    Returns which side of the oriented line the point lies on:
      +1: Left side of directed vector p1 -> p2 (Side A)
      -1: Right side of directed vector p1 -> p2 (Side B)
       0: Exactly colinear
    """
    cross = (line.p2.x - line.p1.x) * (point.y - line.p1.y) - \
            (line.p2.y - line.p1.y) * (point.x - line.p1.x)
    if cross > 1e-4:
        return 1
    elif cross < -1e-4:
        return -1
    return 0


def bottom_center(bbox: Tuple[float, float, float, float]) -> Point:
    """
    Returns bottom-center of bounding box (xmin, ymin, xmax, ymax).
    Best ground-plane anchor point for person feet crossing boundaries.
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    return Point(cx, float(ymax))


def centroid(bbox: Tuple[float, float, float, float]) -> Point:
    """Returns center point (cx, cy) of bounding box."""
    xmin, ymin, xmax, ymax = bbox
    return Point((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance in pixel space."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """
    Width / Height ratio:
    - Normal standing person: ~0.3 - 0.5
    - Crawling / prone person: > 0.85
    """
    xmin, ymin, xmax, ymax = bbox
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    if h <= 1e-5:
        return 0.0
    return w / h


def point_in_polygon(point: Point, polygon_points: np.ndarray) -> bool:
    """Checks if point is inside polygon using cv2.pointPolygonTest."""
    pts = np.asarray(polygon_points, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (point.x, point.y), False) >= 0


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in degrees between two 2D vectors in range [0, 180]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)))


def point_to_segment_distance(point: Point, line: Line) -> float:
    """
    Calculates the shortest Euclidean distance from a point to a finite line segment (p1 -> p2).
    """
    px, py = point.x, point.y
    x1, y1 = line.p1.x, line.p1.y
    x2, y2 = line.p2.x, line.p2.y
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(math.hypot(px - proj_x, py - proj_y))
