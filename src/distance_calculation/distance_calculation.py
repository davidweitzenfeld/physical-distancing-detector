import numpy as np
import itertools
import cv2


def transform_to_ground_plane(points: np.ndarray,
                              ground_plane_homography: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 2

    transformed_points = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2),
                                                  ground_plane_homography).reshape(-1, 2)

    assert transformed_points.shape == points.shape
    return transformed_points


def calculate_distances(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 2

    point_count = len(points)
    combinations = itertools.combinations(range(point_count), 2)

    distances = np.zeros((point_count, point_count))
    for i, j in combinations:
        distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])

    assert distances.shape == (point_count, point_count)
    return distances


def get_points_from_bounding_boxes(bounding_boxes: np.ndarray) -> np.ndarray:
    assert bounding_boxes.ndim == 2 and bounding_boxes.shape[1] == 4

    box_count = len(bounding_boxes)

    points = np.zeros((box_count, 2))
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        points[i, :] = x + (w / 2), y + h  # Bottom middle of bounding box.

    assert points.shape == (box_count, 2)
    return points.astype(int)
