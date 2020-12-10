from typing import Literal, Generator, Callable

import cv2
import numpy as np

ROOT = '../data'

ImageSeq = Generator[np.ndarray, None, None]
ImageSeqProvider = Callable[[], ImageSeq]


def get_grand_central_images() -> ImageSeq:
    return get_video_images(f'{ROOT}/grand_central/grand_central.avi')


def get_oxford_town_center_images() -> ImageSeq:
    return get_video_images(f'{ROOT}/oxford_town_center/oxford_town_center.mp4')


def get_pets_2009_images(subset: Literal[0, 1, 2, 3],
                         difficulty_level: Literal[1, 2, 3],
                         view: Literal[1, 2, 3, 4]) -> ImageSeq:
    for i in range(436):
        path = f'{ROOT}/pets_2009/' \
               f'S{subset}/L{difficulty_level}/Time_14-55/View_{view:03}/frame_{i:04}.jpg'
        yield cv2.imread(path)


def get_virat_v2_ground_images(video_id: str) -> ImageSeq:
    path = f'{ROOT}/virat/Public Dataset/VIRAT Video Dataset Release 2.0/VIRAT Ground Dataset' \
           f'/videos_original/VIRAT_S_{video_id}.mp4'
    return get_video_images(path)


VIRAT_V2_GROUND_000001 = '000001'
VIRAT_V2_GROUND_000101 = '000101'
VIRAT_V2_GROUND_000201_00_000018_000380 = '000201_00_000018_000380'
VIRAT_V2_GROUND_010000_00_000000_000165 = '010000_00_000000_000165'
VIRAT_V2_GROUND_010208_10_000904_000991 = '010208_10_000904_000991'


def get_video_images(path: str) -> ImageSeq:
    video = cv2.VideoCapture(path)
    while video.isOpened():
        _, img = video.read()
        yield img


def get_image_as_seq(path: str) -> ImageSeq:
    yield cv2.imread(path)


def provider(fn: ImageSeqProvider) -> ImageSeqProvider:
    return fn
