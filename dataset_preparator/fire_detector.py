from PIL import Image
from typing import Any, List, Tuple, Optional
from multiprocessing import Process

import cv2
import numpy as np
import os
import gc
import time

from classJson import JsonData
from preparator import ImageDetectionController, DetectionPlace, FIRE_LABEL, LABELS
from preparator import QueuePutter, QueueGetter


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

    def __init__(self,
                 getter: QueueGetter,
                 putter: QueuePutter,
                 fire_threshold: float,
                 iou_threshold: float,
                 path_to_yolo: str,
                 path_to_weights: str,
                 fire_detector: FireDetector):
        super().__init__()
        self._getter = getter
        self._putter = putter
        self._fire_threshold = fire_threshold
        self._iou_threshold = iou_threshold
        self._fire_detector = fire_detector

        self._path_to_yolo = path_to_yolo
        self._path_to_weights = path_to_weights

    @staticmethod
    def _iou(vanilla_box: List[int],
             fire_box: List[int]) -> float:
        intersection_left = max(vanilla_box[1], fire_box[1])
        intersection_right = min(vanilla_box[3], fire_box[3])
        intersection_top = min(vanilla_box[2], fire_box[2])
        intersection_bot = min(vanilla_box[0], fire_box[0])

        intersection = (intersection_right - intersection_left) * (intersection_top - intersection_bot)
        vanilla_square = ((vanilla_box[2] - vanilla_box[0]) * (vanilla_box[3] - vanilla_box[1]))
        fire_square = ((fire_box[2] - fire_box[0]) * (fire_box[3] - fire_box[1]))
        union = vanilla_square + fire_square - intersection

        iou = intersection / union
        return iou

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
        """
        for fire_box in fire_detections:
            for vanilla_box in vanilla_detections:
                iou = self._iou(vanilla_box.detection, fire_box.detection)
                if vanilla_box.label_idx != 2 and iou >= self._iou_threshold:
                    vanilla_box.label = FIRE_LABEL
                    vanilla_box.label_idx = 2
        return vanilla_detections

    def run(self):
        import torch
        model = torch.hub.load(self._path_to_yolo,
                               "custom",
                               self._path_to_weights,
                               force_reload=True,
                               source="local")
        self._fire_detector.set_model(model=model)
        while True:
            print("Fire::Awaiting get")
            value = self._getter.get()
            if not value:
                self._putter.put(None)
                print("Fire:: Got None value, vanilla model is finished, finish fire detector!")
                return
            image_name, image, vanilla_detections = value
            print(f"Fire:: Get {len(vanilla_detections)} vanilla detections.")
            model_time = time.time()
            fire_detections = self._fire_detector.apply_model(image=image,
                                                              threshold=self._fire_threshold)
            print(f"Fire:: Apply fire model for {time.time() - model_time} seconds")

            filter_time = time.time()
            filtered_detections = self._filter_vanilla_detections_by_fire_detections(
                vanilla_detections=vanilla_detections,
                fire_detections=fire_detections
            )
            print(f"Fire:: Filter fire for {time.time() - filter_time} seconds")
            print(f"Fire:: Put {len(filtered_detections)} filtered detections ")
            self._putter.put(value=(image_name, image, filtered_detections))

