from PIL import Image
from typing import Any, List, Tuple, Optional, Set
from logging import Logger
from multiprocessing import Process

import cv2
import numpy as np
import os
import gc
import time
import random
import logging

from classJson import JsonData
from preparator import ImageDetectionController, DetectionPlace, FIRE_LABEL, LABELS, BACKGROUND_LABEL
from preparator import QueuePutter, QueueGetter, plot_detections


class FireDetector:
    _model: Any
    _detection_controller: Optional[ImageDetectionController]

    def __init__(self,
                 detection_controller: Optional[ImageDetectionController] = None):
        self._detection_controller = detection_controller

    def set_model(self, model: Any):
        self._model = model

    def _find_top_best_boxes(self,
                             detections: Any,
                             threshold: float = 0.01) -> List[Tuple[Any, Any]]:
        detections = detections.to("cpu")
        filtered_detections = detections[detections[:, 4] >= threshold]
        scores = [float(x) for x in filtered_detections[:, 4]]
        boxes = []
        for box in filtered_detections[:, :4].numpy():
            boxes.append(
                [int(box[1]), int(box[0]), int(box[3]), int(box[2])]
            )
        return list(zip(scores, boxes))

    def apply_model(self,
                    image: np.ndarray,
                    threshold: float = 0.01) -> List[DetectionPlace]:
        detections = self._model(image)
        if not detections.xyxy[0].shape[0]:
            return []

        top_best_boxes = self._find_top_best_boxes(
            detections=detections.xyxy[0],
            threshold=threshold
        )
        return [
            DetectionPlace(
                place="",
                detection=list(box),
                score=score,
                label=FIRE_LABEL,
                label_idx=2
            )
            for score, box in top_best_boxes
        ]


class MPFireDetectorFilter(Process):
    _getter: QueueGetter
    _putter: QueuePutter

    _fire_detector: FireDetector
    _path_to_yolo: str
    _path_to_weights: str

    _fire_threshold: float
    _iou_threshold: float
    _background_prob_threshold: float

    _logger: Logger

    _debug_image: bool

    def __init__(self,
                 logger: Logger,
                 getter: QueueGetter,
                 putter: QueuePutter,
                 fire_threshold: float,
                 iou_threshold: float,
                 path_to_yolo: str,
                 path_to_weights: str,
                 fire_detector: FireDetector,
                 background_prob_threshold: float = 1.0,
                 need_debug_image: bool = False):
        super().__init__()
        self._getter = getter
        self._putter = putter
        self._fire_threshold = fire_threshold
        self._iou_threshold = iou_threshold
        self._fire_detector = fire_detector

        self._path_to_yolo = path_to_yolo
        self._path_to_weights = path_to_weights

        self._background_prob_threshold = background_prob_threshold

        self._logger = logger

        self._debug_image = need_debug_image

    @staticmethod
    def _iou(vanilla_box: List[int],
             fire_box: List[int]) -> float:
        intersection_left = max(vanilla_box[1], fire_box[1])
        intersection_bot = max(vanilla_box[0], fire_box[0])
        intersection_right = min(vanilla_box[3], fire_box[3])
        intersection_top = min(vanilla_box[2], fire_box[2])
        if intersection_left >= intersection_right or intersection_bot >= intersection_top:
            return 0.0
        intersection = (intersection_right - intersection_left) * (intersection_top - intersection_bot)
        vanilla_square = ((vanilla_box[2] - vanilla_box[0]) * (vanilla_box[3] - vanilla_box[1]))
        iou = intersection / vanilla_square
        return iou

    def _need_to_save_background_image(self) -> bool:
        rand = random.random()
        return rand <= self._background_prob_threshold

    def _filter_background_with_probs(
        self,
        filtered_detections: List[DetectionPlace]
    ) -> List[DetectionPlace]:
        background_to_remove = []
        for idx, place in reversed(list(enumerate(filtered_detections))):
            if place.label_idx == 0 and not self._need_to_save_background_image():
                background_to_remove.append(idx)
        for idx in background_to_remove:
            del filtered_detections[idx]
        return filtered_detections

    def _gen_iou_matrix(
        self,
        vanilla_detections: List[DetectionPlace],
        fire_detections: List[DetectionPlace]
    ) -> np.ndarray:
        iou_matrix = np.zeros((len(fire_detections), len(vanilla_detections)))
        for f_idx, f in enumerate(fire_detections):
            for v_idx, v in enumerate(vanilla_detections):
                iou = self._iou(v.detection, f.detection)
                iou_matrix[f_idx, v_idx] = iou
        return iou_matrix

    def _find_lowest_vanilla_box_for_fire_box(
        self,
        fire_iou: np.ndarray,
        vanilla_detections: List[DetectionPlace],
    ) -> Optional[int]:
        tupled_boxes = [
            (
                v_idx, v_iou, vanilla_detections[v_idx].detection[0]
            )
            for v_idx, v_iou in enumerate(fire_iou) if v_iou > self._iou_threshold]
        if not tupled_boxes:
            return None
        tupled_boxes.sort(key=lambda x: -x[2])
        for t_idx, (v_idx, v_iou, v_box) in enumerate(tupled_boxes):
            vanilla_detections[v_idx].line_thickness = 2 + t_idx
        return tupled_boxes[0][0]

    @staticmethod
    def _handle_lowest_vanilla_box_for_fire(vanilla_box: DetectionPlace):
        if vanilla_box.label_idx == 2:
            # метка уже огонь, все ок
            return
        vanilla_box.label_idx = 2
        vanilla_box.label = FIRE_LABEL

    @staticmethod
    def _handle_not_lowest_vanilla_box_for_fire(vanilla_box: DetectionPlace):
        if vanilla_box.label_idx != 2:
            # метка уже не огонь, все ок
            return
        # считаем backgroundом верхние части языка пламени
        vanilla_box.label_idx = 0
        vanilla_box.label = BACKGROUND_LABEL

    def _filter_vanilla_detections_by_fire_detections(self,
                                                      vanilla_detections: List[DetectionPlace],
                                                      fire_detections: List[DetectionPlace]) -> List[DetectionPlace]:
        """
         Тут идет фильтрация по огню. Какая идея:
         если у ванильном бокса БОЛЬШОЕ (по iou_threshold) пересечение с fire боксом, то:
            1) Если метка ванильного бокса НЕ огонь, то переразмечаем на огонь.
            2) Если метка огонь, все ок.
        если НЕБОЛЬШОЕ пересечение:
            1) То верим ванильной модели
        iou_threshold определяется экспериментально.
        Дополнительно происходит фильтрация пламени, которая заходит на боксы, которые по факту
        не извергают пламя. Идея - искать источник, а не огонь
        """
        iou_matrix = self._gen_iou_matrix(vanilla_detections, fire_detections)
        for f_idx, f in enumerate(fire_detections):
            fire_iou: np.ndarray = iou_matrix[f_idx]
            lowest_vanilla_box_idx = self._find_lowest_vanilla_box_for_fire_box(fire_iou, vanilla_detections)
            for v_idx, iou in np.ndenumerate(fire_iou):
                v_idx = v_idx[0]
                if iou == 0:
                    continue
                if lowest_vanilla_box_idx is not None and v_idx == lowest_vanilla_box_idx:
                    self._handle_lowest_vanilla_box_for_fire(vanilla_box=vanilla_detections[v_idx])
                else:
                    self._handle_not_lowest_vanilla_box_for_fire(vanilla_box=vanilla_detections[v_idx])
        return vanilla_detections

    def debug_draw_image(
            self,
            dir_to_save: str,
            name: str,
            image_np: np.ndarray,
            detections: List[DetectionPlace]
    ):
        CATEGORY_INDEX = {
            label_idx: {"id": label_idx, "name": label}
            for label_idx, label in enumerate(LABELS)
        }
        plot_detections(
            image_np=image_np,
            boxes=np.array([np.array(box.detection) for box in detections]),
            classes=np.array([box.label_idx for box in detections]),
            scores=np.array([box.score for box in detections]),
            category_index=CATEGORY_INDEX,
            image_name=os.path.join(dir_to_save, name),
            line_thickness=[box.line_thickness for box in detections],
            use_normalized_coordinates=False
        )

    def run(self):
        with open("fire_logs.txt", "w") as fl:
            import torch
            model = torch.hub.load(self._path_to_yolo,
                                   "custom",
                                   self._path_to_weights,
                                   force_reload=True,
                                   source="local")
            self._fire_detector.set_model(model=model)
            while True:
                self._logger.info("Fire::Awaiting get")
                value = self._getter.get()
                if not value:
                    self._putter.put(None)
                    self._logger.info("Fire:: Got None value, vanilla model is finished, finish fire detector!")
                    return
                image_name, image, vanilla_detections = value
                self._logger.debug(f"Fire:: Get {len(vanilla_detections)} vanilla detections.")
                model_time = time.time()
                fire_detections = self._fire_detector.apply_model(image=image,
                                                                  threshold=self._fire_threshold)
                self._logger.debug(f"Fire:: Apply fire model for {time.time() - model_time} seconds")

                filter_time = time.time()
                try:
                    filtered_detections = self._filter_vanilla_detections_by_fire_detections(
                        vanilla_detections=vanilla_detections,
                        fire_detections=fire_detections
                    )
                    filtered_detections = self._filter_background_with_probs(filtered_detections=filtered_detections)
                except Exception as e:
                    self._logger.error(e)
                    continue
                if self._debug_image:
                    self.debug_draw_image(
                        dir_to_save=os.environ["DATASET_RESULT_DIR"],
                        name=f"debug_{image_name}",
                        image_np=image,
                        detections=filtered_detections + fire_detections,
                    )
                self._logger.debug(f"Fire:: Filter fire for {time.time() - filter_time} seconds")
                self._logger.debug(f"Fire:: Put {len(filtered_detections)} filtered detections ")
                self._putter.put(value=(image_name, image, filtered_detections))

