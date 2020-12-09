from typing import Tuple

import cv2
import numpy as np


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

    points = np.array(points)

    top_most, bottom_most = np.argmin(points[:, 1]), np.argmax(points[:, 1])
    left_most, right_most = np.argmin(points[:, 0]), np.argmax(points[:, 0])

    points = np.vstack([points[bottom_most, :], points[left_most, :],
                        points[top_most, :], points[right_most, :]])

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


def apply_ground_plane_transform(img: np.ndarray, rectangle: np.ndarray,
                                 unit_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the ground plane transform using the given rectangle unit distances.

    :return: The ground plane transformed image.
    """
    assert unit_dist.shape == (2,)

    h, w = img.shape[:2]
    out_w, out_h = int(w), int(h * (float(unit_dist[0]) / float(unit_dist[1])))
    homography, mask = cv2.findHomography(rectangle, get_corresponding_points(out_w, out_h))
    img_warped = cv2.warpPerspective(img, homography, (out_w, out_h))
    return img_warped, homography


def get_corresponding_points(w: int, h: int) -> np.ndarray:
    return np.array([
        [w, h],  # bottom left
        [0, h],  # top left
        [0, 0],  # top right
        [w, 0],  # bottom right
    ])


if __name__ == '__main__':
    test()
