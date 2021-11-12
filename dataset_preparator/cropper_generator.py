import attr
from PIL import Image
from typing import List, Dict, Set, Tuple, Optional, Any
from multiprocessing import Process
from logging import Logger

import cv2
import json
import os
import pandas as pd
import numpy as np
import uuid
import random
import tqdm
import glob
import time

from classJson import JsonData
from preparator import DetectionPlace, DatasetDirectoryController
from preparator import LABELS, EMISSION_LABEL, FIRE_LABEL
from preparator import QueueGetter

EXTENSION = "jpg"
THRESHOLD = 0.999

SQUARE_DIM = 120


class Cropper:
    _dataset_dir_controller: DatasetDirectoryController

    def __init__(self,
                 dataset_main_dir: str):
        self._dataset_dir_controller = DatasetDirectoryController(dataset_main_dir=dataset_main_dir)
        self._dataset_dir_controller.prepare_directories()

    @staticmethod
    def _increase_box_params(cor1: int,
                             cor2: int,
                             max_value: int,
                             coef: float) -> Tuple[int, int]:
        diff = np.abs(cor1 - cor2)
        min_cor = max(0, cor1 - int(coef * diff))
        max_cor = min(max_value, cor2 + int(coef * diff))
        return min_cor, max_cor

    @staticmethod
    def _increase_square_params(
            c_less: int,
            c_grt: int,
            max_value: int
    ) -> Tuple[int, int]:
        add_diff = (SQUARE_DIM - (c_grt - c_less)) // 2
        less_diff = c_less - add_diff
        grt_diff = max_value - (c_grt + add_diff)

        if less_diff < 0:
            # эту разницу надо добавить к inc_c_grt
            inc_c_less = 0
            inc_c_grt = c_grt + add_diff + np.abs(less_diff)
        elif grt_diff < 0:
            # эту разницу надо добавить к inc_c_less
            inc_c_grt = max_value
            inc_c_less = c_less - add_diff - np.abs(grt_diff)
        else:
            inc_c_less = c_less - add_diff
            inc_c_grt = c_grt + add_diff

        if inc_c_grt - inc_c_less < SQUARE_DIM:
            if inc_c_grt + 1 >= max_value:
                inc_c_less -= 1
            else:
                inc_c_grt += 1

        return inc_c_less, inc_c_grt

    @staticmethod
    def _expand_box(max_height: int,
                    max_width: int,
                    place: DetectionPlace) -> DetectionPlace:
        new_min_y, new_max_y = Cropper._increase_square_params(place.detection[0],
                                                               place.detection[2],
                                                               max_value=max_height)
        new_min_x, new_max_x = Cropper._increase_square_params(place.detection[1],
                                                               place.detection[3],
                                                               max_value=max_width)
        place.square_detection = [new_min_y, new_min_x, new_max_y, new_max_x]
        return place

    @staticmethod
    def _crop_by_place_and_save(dir_controller: DatasetDirectoryController,
                                img: np.ndarray,
                                place: DetectionPlace,
                                img_name: Optional[str] = None):
        if place.score < THRESHOLD:
            return
        cropped_image = img[
            place.square_detection[0]: place.square_detection[2],
            place.square_detection[1]: place.square_detection[3]
        ]
        img_name_without_extenstion = os.path.splitext(img_name)[0]
        croopped_img_name = f"{img_name_without_extenstion}_{uuid.uuid4()}.{EXTENSION}" if img_name else f"{uuid.uuid4()}.{EXTENSION}"
        path_to_save = os.path.join(
            dir_controller.get_directory_for_label_idx(place.label_idx),
            croopped_img_name
        )
        place.image_name = path_to_save
        Image.fromarray(cropped_image).save(path_to_save)

    @staticmethod
    def _get_orig_file_name(image_path: str) -> str:
        splitted = image_path.split('_')
        return f"{splitted[0]}/{splitted[1]}_{splitted[2]}"

    def handle_image(self,
                     img: np.ndarray,
                     places: List[DetectionPlace],
                     img_name: Optional[str] = None):
        for place in places:
            max_height, max_width = img.shape[0], img.shape[1]
            place = Cropper._expand_box(place=place,
                                        max_height=max_height,
                                        max_width=max_width)
            place.original_image_name = Cropper._get_orig_file_name(image_path=img_name)
            Cropper._crop_by_place_and_save(
                dir_controller=self._dataset_dir_controller,
                img=img,
                img_name=img_name,
                place=place
            )


class MPCropperDatasetGenerator(Process):
    _getter: QueueGetter
    _cropper: Cropper
    _path_to_save_csv: str
    _logger: Logger

    def __init__(self,
                 getter: QueueGetter,
                 cropper: Cropper,
                 logger: Logger
                 ):
        self._getter = getter
        self._cropper = cropper
        self._logger = logger
        super().__init__()

    def run(self) -> None:
        while True:
            value = self._getter.get()
            if not value:
                self._logger.info("Cropper:: Got None value from FireDetectorFilter, finish process!")
                # self._cropper.save_pd()
                break
            image_name, image, detections = value
            self._logger.debug(f"Cropper::run:: Get {len(detections)} from q")
            handling_time = time.time()
            try:
                self._cropper.handle_image(img=image, places=detections, img_name=image_name)
            except Exception as e:
                self._logger.error(e)
                continue
            self._logger.debug(f"Handling time {time.time() - handling_time}")


def print_labels_stats(path_to_dataset: str, extension: str = EXTENSION):
    for label in LABELS:
        pattern = f"{os.path.join(path_to_dataset, label)}/*.{extension}"
        print(f"Total {len(glob.glob(pattern))} for label {label}.")

