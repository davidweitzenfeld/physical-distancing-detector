from math import floor
from typing import Tuple, Union

import cv2
import numpy as np
import itertools

from src.utils import clr, data


def test():
    # img = cv2.imread(f'../../data/individual/field.png')
    img = cv2.imread(f'../../data/pets2009/S2/L2/Time_14-55/View_001/frame_0000.jpg')

    rect = ask_for_rectangle(img)
    img_warped = apply_ground_plane_transform(img, rect, (1, 1))
    unit_dist = ask_for_unit_distance(img_warped)
    img_warped = apply_ground_plane_transform(img, rect, unit_dist)

    cv2.imshow('Warped', img_warped)
    cv2.waitKey(0)


def ask_for_rectangle(img: np.ndarray) -> np.ndarray:
    """
    Asks the user to select four points to form a rectangle for the ground plane transformation.

    :return: A 4x2 matrix, where each row represents a point (x,y) of the rectangle
    in order of [bottom-most, left-most, top-most, right-most].
    """
    points = []

    def click_listener(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONUP and len(points) < 4:

            points += [(x, y)]
            cv2.circle(img, (x, y), 10, (0, 255, 0), thickness=2)
            cv2.imshow('Select', img)
            if len(points) == 4:
                points = np.array(points)
                points = order_rectangle(points)
                for i in range(len(points)):
                    cv2.line(img, tuple(points[i]), tuple(points[(i + 1) % len(points)]),
                             clr.blue, thickness=2)
                cv2.imshow('Select', img)

    cv2.namedWindow('Select')
    cv2.setMouseCallback('Select', click_listener)
    cv2.imshow('Select', img)

    done = False
    while not done:
        key = cv2.waitKey()
        if key == ord('r') and len(points) > 0:
            points.pop()
        elif key == 13 and len(points) == 4:  # Enter key
            done = True

    cv2.destroyWindow('Select')

    return points


def ask_for_unit_distance(img: np.ndarray) -> np.ndarray:
    """
    Asks the user to select three points in order to calculate the unit distance.

    :return: A tuple of (x unit distance, y unit distance) in pixels.
    """
    points = []

    def click_listener(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONUP and len(points) < 3:
            if len(points) == 0:  # Reference point
                pass
            elif len(points) == 1:  # X axis point
                y = points[0][1]
            elif len(points) == 2:  # Y axis point
                x = points[0][0]

            points += [(x, y)]
            cv2.circle(img, (x, y), 10, (0, 0, 255), thickness=2)
            cv2.imshow('Select', img)

    cv2.namedWindow('Select')
    cv2.setMouseCallback('Select', click_listener)
    cv2.imshow('Select', img)

    done = False
    while not done:
        key = cv2.waitKey()
        if key == ord('r'):
            points.pop()
        elif key == 13 and len(points) == 3:  # Enter key
            done = True

    cv2.destroyAllWindows()

    x_unit_dist = np.abs(points[0][0] - points[1][0])
    y_unit_dist = np.abs(points[0][1] - points[2][1])
    return np.array([x_unit_dist, y_unit_dist])


def order_rectangle(rectangle: np.ndarray) -> np.ndarray:
    assert rectangle.shape == (4, 2)
    barycenter = np.mean(rectangle, axis=0)
    barycentric = rectangle - barycenter
    sorted_idx = np.argsort(angles(barycentric))
    return rectangle[sorted_idx, :]


def angles(rectangle: np.ndarray) -> np.ndarray:
    assert rectangle.shape == (4, 2)
    barycenter = np.mean(rectangle, axis=0)
    barycentric = rectangle - barycenter
    return np.array([angle(b) for b in barycentric])


def angle(v: np.ndarray) -> float:
    a = np.arctan2(*v[::-1])
    a = np.degrees(a)
    return int((a - 90) % 360)


def compute_homography(img: np.ndarray, rectangle: np.ndarray,
                       unit_dist: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Computes the homography for the ground plane transform
    using the given rectangle and unit distances.

    :return: A tuple of (ground plane transform homography, size of the output image).
    """
    assert unit_dist.shape == (2,)
    h, w = img.shape[:2]

    left_offset = np.min(rectangle[:, 0])
    right_offset = w - np.max(rectangle[:, 0])
    top_offset = np.min(rectangle[:, 1])
    bottom_offset = h - np.max(rectangle[:, 1])
    translation = np.array([[1, 0, left_offset], [0, 1, top_offset], [0, 0, 1]])

    homog_w, homog_h = int(w), int(h * (float(unit_dist[0]) / float(unit_dist[1])))

    corners = get_corners(homog_w, homog_h)
    homography, mask = cv2.findHomography(rectangle, corners)

    out_w, out_h = homog_w + left_offset + right_offset, homog_h + top_offset + bottom_offset
    return translation @ homography, (out_w, out_h)


def apply_ground_plane_transform(img: np.ndarray, homography: np.ndarray,
                                 out_size: Tuple[int, int]) -> np.ndarray:
    """
    Applies the ground plane transform homography.

    :return: The ground plane transformed image.
    """
    img_warped = cv2.warpPerspective(img, homography, out_size)
    return img_warped


def get_corners(w: int, h: int) -> np.ndarray:
    return np.array([
        [0, h],  # bottom left
        [0, 0],  # top left
        [w, 0],  # top right
        [w, h],  # bottom right
    ])


def get_ground_plane_img(images: data.ImageSeq, frame_count: Union[int, None],
                         homography: np.ndarray,
                         out_size: Tuple[int, int], sample_size: int = 500) -> np.ndarray:
    if frame_count is not None:
        sample_size = min(frame_count, sample_size)
        step = int(floor(frame_count / sample_size))
    else:
        step = 1
    matrix = np.stack([cv2.warpPerspective(img, homography, out_size)
                       for img in itertools.islice(images, 0, sample_size, step)])
    mean = np.mean(matrix, axis=0)
    mean = cv2.convertScaleAbs(mean)
    return mean


if __name__ == '__main__':
    test()
