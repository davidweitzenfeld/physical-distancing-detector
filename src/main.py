from typing import Generator

from src.distance_calculation.distance_calculation import *
from src.ground_plane_transformation.homography import *
from src.pedestrian_detection.yolo import *
from src.utils import clr, data


def main():
    video_win = "video"
    ground_win = "ground"

    image_provider = data.provider(lambda: data.get_pets_2009_images(2, 2))

    for img, img_ground in process(image_provider):
        cv2.imshow(video_win, img)
        cv2.imshow(ground_win, img_ground)
        cv2.waitKey(1)


def ask_for_preparation_data(images_provider: data.ImageSeqProvider) \
        -> Tuple[np.ndarray, np.ndarray]:
    images = images_provider()
    img = next(images)

    rect = ask_for_rectangle(img)
    homography, ground_size = compute_homography(img, rect, np.ones(2, ))
    img_ground = apply_ground_plane_transform(img, homography, ground_size)
    unit_dist = ask_for_unit_distance(img_ground)

    assert rect.shape == (4, 2) and unit_dist.shape == (2,)
    return rect, unit_dist


def process(images_provider: data.ImageSeqProvider,
            rect: np.ndarray, unit_dist: np.ndarray) \
        -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    images = images_provider()
    img = next(images)

    # Ground plane transformation.
    print('Performing ground plane transformation...')
    homography, ground_size = compute_homography(img, rect, unit_dist)
    img_ground = get_ground_plane_img(images_provider(), None, homography, ground_size)
    print('Done!')

    # Pedestrian detection.
    print('Preparing pedestrian detection...')
    net, last_layers, label_names = prepare_yolo_model('../data/yolo_v3_coco')
    print('Done!')

    for img in images:

        # Pedestrian detection.
        print('Performing pedestrian detection...')
        bounding_boxes, confidences = detect_people(img, net, last_layers, label_names)
        print('Done!')

        # Distance and other calculations.
        print('Calculating...')
        points = get_points_from_bounding_boxes(bounding_boxes)
        points_ground = transform_to_ground_plane(points, homography)
        distances = calculate_distances(points_ground / unit_dist)
        print('Done!')

        non_zero_dist = np.ma.masked_array(distances, mask=distances == 0)
        smallest_dist_per_point = np.min(non_zero_dist, axis=1).T.reshape(-1, 1)

        for i, (x, y, w, h) in enumerate(bounding_boxes):
            dist = smallest_dist_per_point[i]
            color = clr.red if dist <= 1 else clr.orange if dist <= 1.6 else clr.green
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)

        img_ground_copy = img_ground.copy()
        for i, (x, y) in enumerate(points_ground):
            dist = smallest_dist_per_point[i]
            color = clr.red if dist <= 1 else clr.orange if dist <= 1.6 else clr.green
            cv2.circle(img_ground_copy, (x, y), 5, color, thickness=2)
        for i, j in np.ndindex(distances.shape):
            if i == j or distances[i, j] > 1:
                continue
            i_x, i_y = points_ground[i]
            j_x, j_y = points_ground[j]
            h_x, h_y = (points_ground[i] + (points_ground[j] - points_ground[i]) / 2).astype(int)
            cv2.line(img_ground_copy, (i_x, i_y), (j_x, j_y), clr.red, thickness=2)
            cv2.putText(img_ground_copy, f'{distances[i, j]:.2f}', (h_x, h_y),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=clr.green)

        yield img, img_ground_copy


if __name__ == '__main__':
    main()
