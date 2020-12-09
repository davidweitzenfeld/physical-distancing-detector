# Implementation of website was initially based on the following guide.
#   https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

import cv2
import flask
import threading

from src.utils import data
from src.main import process

output_videos = (None, None)
lock = threading.Lock()

app = flask.Flask(__name__, template_folder='web', static_folder='web')


@app.route('/')
def index():
    # noinspection PyUnresolvedReferences
    return flask.render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return flask.Response(stream_out(idx=0),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ground_feed')
def ground_feed():
    return flask.Response(stream_out(idx=1),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    global output_videos, lock

    images_provider = data.provider(lambda: data.get_pets_2009_images(2, 2))
    for img, ground in process(images_provider):
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
    thread = threading.Thread(target=main)
    thread.daemon = True
    thread.start()

    app.run(threaded=True)
