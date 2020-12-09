import cv2
from src.ground_plane_transformation.homography import *
from src.pedestrian_detection.yolo import *
from src.distance_calculation.distance_calculation import *
from src.utils import clr


def main():
    video_win = "video"
    ground_win = "ground"

    images = (f'../data/pets2009/S2/L2/Time_14-55/View_001/frame_{n:04}.jpg' for n in range(200))

    img = cv2.imread(f'../data/pets2009/S2/L2/Time_14-55/View_001/frame_0000.jpg')

    # Ground plane transformation.
    print('Performing ground plane transformation...')
    rect = ask_for_rectangle(img)
    img_ground, _ = apply_ground_plane_transform(img, rect, np.ones((2,)))
    unit_dist = ask_for_unit_distance(img_ground)
    img_ground, homography = apply_ground_plane_transform(img, rect, unit_dist)
    print('Done!')

    # Pedestrian detection.
    print('Preparing pedestrian detection...')
    net, last_layers, label_names = prepare_yolo_model('../data/yolo_v3_coco')
    print('Done!')

    for img_path in images:
        img = cv2.imread(img_path)

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
            color = clr.red if dist <= 1 else clr.green
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
        for x, y in points:
            cv2.circle(img, (x, y), 5, clr.blue, thickness=3)

        img_ground_copy, _ = apply_ground_plane_transform(img, rect, unit_dist)
        for x, y in points_ground:
            cv2.circle(img_ground_copy, (x, y), 5, clr.blue, thickness=2)
        for i, j in np.ndindex(distances.shape):
            if i == j or distances[i, j] > 1:
                continue
            i_x, i_y = points_ground[i]
            j_x, j_y = points_ground[j]
            h_x, h_y = (points_ground[i] + (points_ground[j] - points_ground[i]) / 2).astype(int)
            cv2.line(img_ground_copy, (i_x, i_y), (j_x, j_y), clr.green, thickness=2)
            cv2.putText(img_ground_copy, f'{distances[i, j]:.2f}', (h_x, h_y),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=clr.green)

        cv2.imshow(video_win, img)
        cv2.imshow(ground_win, img_ground_copy)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
