# Implementation of website was initially based on the following guide.
#   https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
from typing import Optional

import cv2
import flask
import threading
import numpy as np

from src.utils import data
from src.main import ask_for_preparation_data, process

images_provider: Optional[data.ImageSeqProvider] = None
rectangle_points: Optional[str] = None
unit_distances: Optional[str] = None
thread: Optional[threading.Thread] = None
stop_thread = False
output_videos = (None, None)
lock = threading.Lock()

app = flask.Flask(__name__, template_folder='web', static_folder='web')

datasets = {
    'grand_central': 'Grand Central',
    'oxford_town_center': 'Oxford Town Center',
    'pets_2009_s2_l2_v1': 'PETS 2009 Subset 2 Level 2 View 1',
    'virat_v2_ground_000101': 'VIRAT Version 2 Ground 000101',
    'virat_v2_ground_010208_10_000904_000991': 'VIRAT Version 2 Ground 010208 10 000904 000991',
}

dataset_images_providers = {
    'grand_central': lambda: data.get_grand_central_images(),
    'oxford_town_center': lambda: data.get_oxford_town_center_images(),
    'pets_2009_s2_l2_v1': lambda: data.get_pets_2009_images(subset=2, difficulty_level=2, view=1),
    'virat_v2_ground_000101': lambda: data.get_virat_v2_ground_images(data.VIRAT_V2_GROUND_000101),
    'virat_v2_ground_010208_10_000904_000991': lambda: data.get_virat_v2_ground_images(
        data.VIRAT_V2_GROUND_010208_10_000904_000991),
}

dataset_preparation_data = {
    'grand_central': 'rectangle=50,429,171,99,552,96,717,434&unit_dist=38,39',
    'oxford_town_center': 'rectangle=0,664,1045,133,1888,243,1474,1065&unit_dist=300,150',
    'pets_2009_s2_l2_v1': 'rectangle=10,286,237,103,767,129,395,379&unit_dist=71,103',
    'virat_v2_ground_000101': 'rectangle=67,488,418,378,1161,665,761,933&unit_dist=545,358',
    'virat_v2_ground_010208_10_000904_000991':
        'rectangle=45,313,434,302,1037,506,758,587&unit_dist=270,199',
}


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def index():
    stop_thread_if_running()

    # noinspection PyUnresolvedReferences
    return flask.render_template('index.html',
                                 datasets=datasets, prep_data=dataset_preparation_data)


@app.route('/preparation')
def preparation():
    dataset_id = flask.request.args.get('dataset')

    if dataset_id is None:
        return flask.redirect('/')

    stop_thread_if_running()
    start_preparation_thread(dataset_id)

    # noinspection PyUnresolvedReferences
    return flask.render_template('preparation.html',
                                 dataset_id=dataset_id, dataset=datasets[dataset_id])


@app.route('/detection')
def detection():
    dataset_id = flask.request.args.get('dataset')
    rect = flask.request.args.get('rectangle')
    unit_dist = flask.request.args.get('unit_dist')

    if dataset_id is None:
        return flask.redirect('/')
    if rect is None or unit_dist is None:
        if rectangle_points is not None and unit_distances is not None:
            return flask.redirect(f'/detection?dataset={dataset_id}'
                                  f'&rectangle={rectangle_points}&unit_dist={unit_distances}')
        else:
            return flask.redirect(f'/preparation?dataset={dataset_id}')

    stop_thread_if_running()
    start_processing_thread(dataset_id, rect, unit_dist)

    # noinspection PyUnresolvedReferences
    return flask.render_template('detection.html', dataset=datasets[dataset_id])


@app.route('/stop')
def stop():
    stop_thread_if_running()
    return flask.redirect('/')


@app.route('/video_feed')
def video_feed():
    return flask.Response(stream_out(idx=0),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ground_feed')
def ground_feed():
    return flask.Response(stream_out(idx=1),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def stop_thread_if_running():
    global thread, stop_thread
    if thread is not None:
        stop_thread = True


def start_preparation_thread(dataset_id: str):
    global images_provider, thread, stop_thread

    stop_thread = False

    images_provider = dataset_images_providers[dataset_id]
    thread = threading.Thread(target=preparation_thread)
    thread.daemon = True
    thread.start()


def start_processing_thread(dataset_id: str, rect: str, unit_dist: str):
    global images_provider, thread, rectangle_points, unit_distances, stop_thread

    stop_thread = False

    images_provider = dataset_images_providers[dataset_id]
    rectangle_points, unit_distances = rect, unit_dist

    thread = threading.Thread(target=processing_thread)
    thread.daemon = True
    thread.start()


def preparation_thread():
    global images_provider, lock, rectangle_points, unit_distances

    if images_provider is None:
        raise Exception()

    rect, unit_dist = ask_for_preparation_data(images_provider)
    with lock:
        rectangle_points = ','.join([str(n) for n in rect.flatten()])
        unit_distances = ','.join([str(n) for n in unit_dist])
    print('Done!')


def processing_thread():
    global images_provider, output_videos, lock, stop_thread, thread

    if images_provider is None or rectangle_points is None or unit_distances is None:
        raise Exception()

    rect = np.array([int(n) for n in rectangle_points.split(',')]).reshape((4, 2))
    unit_dist = np.array([int(n) for n in unit_distances.split(',')]).reshape((2,))

    for img, ground in process(images_provider, rect, unit_dist):
        if stop_thread:
            print('Stopping thread.')
            stop_thread = False
            thread = None
            break
        with lock:
            output_videos = img.copy(), ground.copy()


def stream_out(idx: int):
    global output_videos, lock

    while True:
        with lock:
            if output_videos[idx] is None:
                continue

            flag, encoded_image = cv2.imencode('.jpg', output_videos[idx])
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


if __name__ == '__main__':
    app.run(threaded=True)
