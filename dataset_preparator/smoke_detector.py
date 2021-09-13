from typing import Any, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from dataset_preparator.preparator import ImageDetectionController, DetectionPlace, EMISSION_LABEL


class SmokeDetector:
    IMG_HEIGHT, IMG_WIDTH = 480, 640

    _model: Any
    _detection_controller: ImageDetectionController

    def __init__(self,
                 path_to_model: str,
                 detection_controller: ImageDetectionController):
        tf.keras.backend.clear_session()
        self._detection_controller = detection_controller
        self._model = tf.saved_model.load(path_to_model)

    def _find_top_5_best_box(self,
                             scores: np.ndarray,
                             boxes: np.ndarray,
                             max_height: int,
                             max_width: int,
                             threshold: float = 0.01) -> List[Tuple[float, List[int]]]:
        good_scores = [float(score) for score in scores[scores >= threshold]]
        good_boxes = []
        for box in boxes[:5]:
            good_boxes.append([
                int(box[0] * max_height),
                int(box[1] * max_width),
                int(box[2] * max_height),
                int(box[3] * max_width)
            ])
        return list(zip(good_scores, good_boxes))

    def handle_image(self,
                     matrix: List[List],
                     max_height: int,
                     max_width: int,
                     image_path: str,
                     threshold: float = 0.01):
        frame: np.ndarray = cv2.imread(image_path)
        frame = cv2.warpPerspective(
            frame,
            np.array(matrix),
            (max_width, max_height),
            flags=cv2.INTER_LINEAR
        )
        frame = np.expand_dims(frame, 0)
        detections = self._model(frame)

        top_5_best_boxes = self._find_top_5_best_box(
            scores=detections["detection_scores"][0].numpy(),
            boxes=detections["detection_boxes"][0].numpy(),
            max_width=max_width,
            max_height=max_height,
            threshold=threshold
        )

        if top_5_best_boxes:
            self._detection_controller.add(
                img_path=image_path,
                img_detections=[
                    DetectionPlace(
                        place="",
                        detection=list(box),
                        score=score,
                        label=EMISSION_LABEL,
                        label_idx=1
                    )
                    for score, box in top_5_best_boxes
                ]
            )