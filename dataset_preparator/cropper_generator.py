from PIL import Image
from typing import List, Dict, Set, Tuple, Optional, Any
from multiprocessing import Process

import cv2
import json
import os
import numpy as np
import uuid
import tqdm
import glob
import time

from classJson import JsonData
from preparator import DetectionPlace, DatasetDirectoryController
from preparator import LABELS, EMISSION_LABEL, FIRE_LABEL
from preparator import QueueGetter

EXTENSION = "png"
THRESHOLD = 0.999


class Cropper:
    _vanilla_model_results: Dict[str, List[Dict[str, Any]]]
    _replacing_patterns: Optional[List[Tuple[str, str]]]
    _dataset_dir_controller: DatasetDirectoryController

    def __init__(self,
                 dataset_main_dir: str,
                 path_to_vanilla_model_results: Optional[str] = None,
                 replacing_patterns: Optional[List[Tuple[str, str]]] = None):
        if path_to_vanilla_model_results:
            with open(path_to_vanilla_model_results) as p:
                self._vanilla_model_results = json.load(p)
        self._replacing_patterns = replacing_patterns
        self._dataset_dir_controller = DatasetDirectoryController(dataset_main_dir=dataset_main_dir)
        self._dataset_dir_controller.prepare_directories()

    def _replace_patterns(self,
                          image_path: str) -> str:
        if not self._replacing_patterns:
            return image_path
        result_path = image_path
        for pattern, result in self._replacing_patterns:
            result_path = result_path.replace(pattern, result)
        return result_path

    @staticmethod
    def _increase_box_params(cor1: int,
                             cor2: int,
                             max_value: int,
                             coef: float) -> Tuple[int, int]:
        diff = np.abs(cor1 - cor2)
        min_cor = max(0, cor1 - int(coef * diff))
        max_cor = min(max_value, cor2 + int(coef * diff))
        return min_cor, max_cor

    def _expand_box(self,
                    max_height: int,
                    max_width: int,
                    place: DetectionPlace) -> DetectionPlace:
        new_min_y, new_max_y = self._increase_box_params(place.detection[0],
                                                         place.detection[2],
                                                         max_value=max_height,
                                                         coef=1.5)
        new_min_x, new_max_x = self._increase_box_params(place.detection[1],
                                                         place.detection[3],
                                                         max_value=max_width,
                                                         coef=1.5)
        place.detection = [new_min_y, new_min_x, new_max_y, new_max_x]
        return place

    def _crop_by_place_and_save(self,
                                img: np.ndarray,
                                place: DetectionPlace,
                                img_name: Optional[str] = None):
        if place.score >= THRESHOLD:
            cropped_image = img[
                place.detection[0]: place.detection[2],
                place.detection[1]: place.detection[3]
            ]
            path_to_save = os.path.join(
                self._dataset_dir_controller.get_directory_for_label_idx(place.label_idx),
                img_name if img_name else f"{uuid.uuid4()}.{EXTENSION}"
            )
            Image.fromarray(cropped_image).save(path_to_save)

    def handle_image(self,
                     img: np.ndarray,
                     places: List[DetectionPlace],
                     img_name: Optional[str] = None):
        for place in places:
            max_height, max_width = img.shape[0], img.shape[1]
            place = self._expand_box(place=place,
                                     max_height=max_height,
                                     max_width=max_width)
            self._crop_by_place_and_save(img=img,
                                         img_name=img_name,
                                         place=place)


class MPCropperDatasetGenerator(Process):
    _getter: QueueGetter
    _cropper: Cropper

    def __init__(self,
                 getter: QueueGetter,
                 cropper: Cropper):
        self._getter = getter
        self._cropper = cropper
        super().__init__()

    def run(self) -> None:
        while True:
            value = self._getter.get()
            if not value:
                print("Cropper:: Got None value from FireDetectorFilter, finish process!")
                break
            image_name, image, detections = value
            print(f"Cropper::run:: Get {len(detections)} from q")
            handling_time = time.time()
            self._cropper.handle_image(img=image, places=detections, img_name=image_name)
            print(f"Handling time {time.time() - handling_time}")


def print_labels_stats(path_to_dataset: str, extension: str = EXTENSION):
    for label in LABELS:
        pattern = f"{os.path.join(path_to_dataset, label)}/*.{extension}"
        print(f"Total {len(glob.glob(pattern))} for label {label}.")

