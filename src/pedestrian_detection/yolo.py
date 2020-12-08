from typing import List, Tuple

import cv2
import numpy as np
import time

YOLO = '../../data/yolo_v3_coco'


# This code is based on the following YOLOv3 processing code:
#   https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
#   https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/

def test():
    net, last_layers, label_names = prepare_yolo_model()

    start = time.time()
    for i in range(100):
        frame = cv2.imread(f'../../data/pets2009/S2/L2/Time_14-55/View_001/frame_{i:04}.jpg')
        boxes, _ = detect_people(frame, net, last_layers, label_names)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        dur = time.time() - start
        t = dur / (i + 1)
        print(f'{1 / t} fps; {t} per frame.')

        cv2.imshow('Pedestrians', frame)
        cv2.waitKey(1)


def prepare_yolo_model() -> Tuple[cv2.dnn_Net, List[str], List[str]]:
    """
    Prepares the YOLOv3 neural network model.

    :return: A tuple of (net, last_layers, label_names).
    """
    # Read label names.
    with open(f'{YOLO}/coco.names') as f:
        label_names = f.read().splitlines()

    # Read the YOLO network model.
    net = cv2.dnn.readNetFromDarknet(f'{YOLO}/yolo_v3.cfg', f'{YOLO}/yolo_v3.weights')

    # Get the output layers of the network.
    # The last layers are those which do not have an output (i.e. "leaf nodes").
    all_layers = net.getLayerNames()
    last_layers = [all_layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, last_layers, label_names


def detect_people(img: np.ndarray, net: cv2.dnn_Net,
                  last_layers: List[str], names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects people in the given img.

    :return: A tuple of (bounding_boxes, confidences).
    """
    img_h, img_w = img.shape[:2]

    # Create normalized and resized blob from image. Also OpenCV's BGR is converted to RGB.
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Feed the image blob through the network, until the last layers.
    net.setInput(blob)
    net_outputs = net.forward(last_layers)

    # Get all object detections from the network outputs.
    detections = [detection for output in net_outputs for detection in output]

    # Filter detections by score and confidence.
    filtered_detections = []
    for detection in detections:
        bounding_box = detection[:4]
        scores = detection[5:]

        # We only care about person objects.
        max_class = np.argmax(scores)
        if max_class != names.index('person'):
            continue

        # We only care if confidence is above a threshold.
        confidence = scores[max_class]
        if confidence < 0.3:
            continue

        filtered_detections += [np.concatenate([bounding_box, [confidence]])]

    # Process the detections.
    bounding_boxes = np.zeros((len(filtered_detections), 4), int)
    confidences = np.zeros((len(filtered_detections)), float)
    for i, detection in enumerate(filtered_detections):
        bounding_box = detection[:4]
        confidence = detection[4]

        # We need to scale the bounding box.
        bounding_box_scaling = np.array([img_w, img_h] * 2)
        bounding_box *= bounding_box_scaling

        # We want to convert the bounding box format.
        center_x, center_y, w, h = bounding_box

        bounding_boxes[i, :] = int(center_x - (w / 2)), int(center_y - (h / 2)), int(w), int(h)
        confidences[i] = float(confidence)

    # Apply non-maxima suppression to remove overlapping boxes with low confidences.
    indexes = cv2.dnn.NMSBoxes(list(bounding_boxes), list(confidences),
                               score_threshold=0.5, nms_threshold=0.5)

    return bounding_boxes[indexes].reshape(-1, 4), confidences[indexes].reshape(-1, 1)


if __name__ == '__main__':
    test()
