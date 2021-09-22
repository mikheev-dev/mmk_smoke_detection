from typing import Any, List, Optional, Iterator, Tuple, Dict

import attr
import cv2
import json
import numpy as np
import os
import random
import re
import time
from PIL import Image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input

from classJson import JsonData
from preparator import RoiLoader, ImageDetectionController, N, M, DetectionPlace, LABELS
from multiprocessing import Process, Queue
from preparator import QueuePutter, AbsGetter
from logging import Logger


#TODO Realize the getter for real time cv2.
class FileGetter(AbsGetter):
    _pathes: Iterator

    def __init__(self,
                 pathes_to_images: List[str]):
        print(len(pathes_to_images))
        self._pathes = iter(pathes_to_images)

    def _gen_image_name(self,
                        image_path: str) -> str:
        return f"{os.path.dirname(image_path).split('/')[-1]}_{os.path.basename(image_path)}"

    def get(self) -> Optional[Tuple[str, np.ndarray]]:
        try:
            image_path = next(self._pathes)
            img = np.array(Image.open(image_path))
            return self._gen_image_name(image_path), img
        except StopIteration:
            return None


class MultiCameraFileGetter(AbsGetter):
    _pathes: Iterator
    _total_count: int
    _counter: int = 0

    _logger: Logger

    def __init__(self,
                 pathes_to_images: List[str],
                 logger: Logger):
        self._logger = logger
        self._logger.info(len(pathes_to_images))
        self._total_count = len(pathes_to_images)
        self._pathes = iter(pathes_to_images)
        self._counter = 0

    def _get_camera_ip(self,
                       image_path: str) -> str:
        return os.path.dirname(image_path).split('/')[-1]

    def _gen_image_name(self,
                        image_path: str) -> str:
        return f"{self._get_camera_ip(image_path)}_{os.path.basename(image_path)}"

    def get(self) -> Optional[Tuple[str, str, np.ndarray]]:
        try:
            image_path = next(self._pathes)
            img = np.array(Image.open(image_path))
            ip = self._get_camera_ip(image_path)
            self._counter += 1
            self._logger.info(f"MultiCameraFileGetter:: Loaded {self._counter}/{self._total_count}")
            return self._gen_image_name(image_path), ip, img
        except StopIteration:
            return None

@attr.s
class CameraConfig:
    matrix: np.ndarray = attr.ib(default=np.array([]))
    max_height: int = attr.ib(default=0)
    max_width: int = attr.ib(default=0)
    roi: Optional[RoiLoader] = attr.ib(default=None)


class CameraConfigController:
    _ip_to_config: Dict[str, CameraConfig]
    
    def __init__(self,
                 configs: List[str]):
        self._ip_to_config = {}
        for cfg_path in configs:
            json_config = JsonData(cfg_path)
            ip = self._get_ip(json_config)
            matrix_data = json_config.return_matrix()
            self._ip_to_config[ip] = CameraConfig(
                matrix=np.array(matrix_data["matrix"]),
                max_height=matrix_data["maxHeight"],
                max_width=matrix_data["maxWidth"],
                roi=RoiLoader(json_config)
            )
            
    def _get_ip(self, data: JsonData) -> str:
        ip_pattern = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
        return re.findall(ip_pattern, data.return_address())[0]
    
    def get_config_by_ip(self,
                         ip: str) -> CameraConfig:
        return self._ip_to_config[ip]


class VanillaDetector:
    _model: Any
    _emission_threshold: Optional[float]
    _detection_controller: Optional[ImageDetectionController]
    _logger: Logger

    def __init__(self,
                 logger: Logger,
                 emission_threshold: Optional[float] = None,
                 detection_controller: Optional[ImageDetectionController] = None) -> None:
        self._emission_threshold = emission_threshold
        self._detection_controller = detection_controller
        self._logger = logger

    def set_model(self, model: Any):
        self._model = model

    @staticmethod
    def _apply_perspective(frame,
                           camera_config: CameraConfig):
        if frame is not None:
            return cv2.warpPerspective(
                frame,
                camera_config.matrix,
                (camera_config.max_width, camera_config.max_height),
                flags=cv2.INTER_LINEAR
            )

    def _prepare_batch(self, frame, camera_config: CameraConfig):
        arr = np.zeros([len(camera_config.roi.coordinates_roi), N, M, 3])
        try:
            for idx in range(len(camera_config.roi.coordinates_roi)):
                r_ = camera_config.roi.coordinates_roi[idx]
                crop_frame = frame[r_['y_min']:r_['y_max'], r_['x_min']:r_['x_max']]
                crop_frame = cv2.resize(crop_frame, (M, N))
                crop_frame = np.reshape(crop_frame, (1, N, M, 3))
                arr[idx] = crop_frame
        except Exception as error:
            self._logger.error(str(error))
        return arr

    def _get_image_classification_idx(self,
                                      predictions: np.ndarray) -> int:
        img_classification_idx = 0
        for prediction in predictions:
            mpc = int(np.argmax(prediction))
            if mpc > img_classification_idx:
                img_classification_idx = mpc
        return img_classification_idx
    
    def _try_to_change_label_to_emission(self,
                                         label_idx: int,
                                         prediction: np.ndarray) -> int:
        if not self._emission_threshold:
            return label_idx
        return 1 if label_idx == 0 and prediction[1] > self._emission_threshold else label_idx

    def _get_detection_places_of_image(self,
                                       predictions: np.ndarray,
                                       camera_config: CameraConfig) -> List[DetectionPlace]:
        detection_places = []
        for place_idx, prediction in enumerate(predictions):
            label_idx = self._try_to_change_label_to_emission(
                prediction=prediction,
                label_idx=int(np.argmax(prediction))
            )
            coors = list(camera_config.roi.coordinates_roi[place_idx].values())
            ordered_coors = [coors[1], coors[0], coors[3], coors[2]]
            detection_places.append(
                DetectionPlace(
                    place=str(place_idx),
                    detection=ordered_coors,
                    label_idx=label_idx,
                    label=LABELS[label_idx],
                    score=float(prediction[label_idx]),
                )
            )
        return detection_places

    def _postprocess_image(self,
                           img_path: str,
                           predictions: List[DetectionPlace]):
        if self._detection_controller:
            self._detection_controller.add(img_path=img_path,
                                           img_detections=predictions)

    def _save_frame(self,
                    img: np.ndarray,
                    path: str):
        Image.fromarray(img).save(path)

    def _generate_image_name(self, image_path: str):
        return "_".join(image_path.split('/')[-2:])

    def apply_model(self,
                    image: np.ndarray,
                    camera_config: CameraConfig) -> Tuple[np.ndarray, List[DetectionPlace]]:
        image = self._apply_perspective(image, camera_config=camera_config)
        batch = self._prepare_batch(image, camera_config=camera_config)
        batch = preprocess_input(batch)
        predictions = self._model.predict_on_batch(batch)
        return image, self._get_detection_places_of_image(
            predictions=predictions,
            camera_config=camera_config
        )


class MPVanillaDetector(Process):
    _getter: AbsGetter
    _putter: QueuePutter

    _vanilla_detector: VanillaDetector
    _config_controller: CameraConfigController
    _path_to_model: str

    _logger: Logger

    def __init__(self,
                 logger: Logger,
                 getter: AbsGetter,
                 putter: QueuePutter,
                 path_to_model: str,
                 vanilla_detector: VanillaDetector,
                 config_controller: CameraConfigController):
        super().__init__()
        self._getter = getter
        self._putter = putter
        self._vanilla_detector = vanilla_detector
        self._path_to_model = path_to_model
        self._config_controller = config_controller
        self._logger = logger

    def _filter_background_by_probs(self,
                                    detections: List[DetectionPlace]) -> List[DetectionPlace]:
        filtered = []
        for det in detections:
            if det.label_idx != 0 or random.random() <= 0.6:
                filtered.append(det)
        return filtered

    def run(self) -> None:
        from tensorflow import keras
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        vanilla_model = keras.models.load_model(self._path_to_model, compile=True)
        self._vanilla_detector.set_model(vanilla_model)
        while True:
            image_value: Optional[Tuple[str, str, np.ndarray]] = self._getter.get()
            if image_value is None:
                self._putter.put(value=None)
                self._logger.info("Vanilla::Can't get new value from getter, finish process!")
                break
            image_name, ip, image = image_value
            model_time = time.time()
            perspective_image, detections = self._vanilla_detector.apply_model(
                image=image,
                camera_config=self._config_controller.get_config_by_ip(ip)
            )
            detections = self._filter_background_by_probs(detections)
            self._logger.debug(f"Vanilla::Get {len(detections)} detections for {time.time() - model_time} seconds")
            if detections:
                self._putter.put(value=(image_name, perspective_image, detections))
                self._logger.debug(f"Vanilla::Put detections to queue")
