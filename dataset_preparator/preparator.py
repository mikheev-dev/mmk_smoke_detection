import json
import attr
import cv2
import numpy as np
import os
import random
import abc

from PIL import Image
from multiprocessing import Queue

from typing import Dict, List, Any, Optional
from classJson import JsonData

PATH_TO_MODEL = "vanilla_model/hatches.h5"
BACKGROUND_LABEL = "background"
EMISSION_LABEL = "emission"
FIRE_LABEL = "fire"
MACHINE_LABEL = "machine"

LABELS = [BACKGROUND_LABEL, EMISSION_LABEL, FIRE_LABEL, MACHINE_LABEL]
N, M = 75, 75  # размер входного изображения


from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    line_thickness=None,
                    figsize=(12, 16),
                    image_name=None,
                    use_normalized_coordinates=True):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    if not line_thickness:
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            category_index,
            line_thickness=4,
            use_normalized_coordinates=use_normalized_coordinates,
            min_score_thresh=0.01)
    else:
        for box, cls, score, thickness in zip(boxes, classes, scores, line_thickness):
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.array([box]),
                np.array([cls]),
                np.array([score]),
                category_index,
                line_thickness=thickness,
                use_normalized_coordinates=use_normalized_coordinates,
                min_score_thresh=0.01)

    if image_name:
        plt.imsave(image_name, image_np)
    else:
        plt.imshow(image_np)


class AbsPutter(abc.ABC):
    @abc.abstractmethod
    def put(self, value: Any):
        raise NotImplementedError


class AbsGetter(abc.ABC):
    @abc.abstractmethod
    def get(self, *args, **kwargs):
        raise NotImplementedError


class QueuePutter(AbsPutter):
    _q: Queue

    def __init__(self,
                 q: Queue):
        self._q = q

    def put(self, value: Any):
        self._q.put(value)


class QueueGetter(AbsGetter):
    _q: Queue

    def __init__(self,
                 q: Queue):
        self._q = q

    def get(self) -> Any:
        return self._q.get()


class RoiLoader:
    coordinates_roi: List
    fake_coordinates_roi: List
    numbers: List
    side: List

    def __init__(self,
                 jd: JsonData):
        coord_machine = jd.return_roi_machine()
        coord_coke = jd.return_roi_coke()
        coord_machine_fake = jd.return_fake_machine()
        coord_coke_fake = jd.return_fake_coke()

        self.coordinates_roi = []
        self.fake_coordinates_roi = []
        self.numbers = []
        self.side = []
        # загрузка координат ROI
        if coord_coke is not None and coord_coke_fake is not None:
            start_end_coke = jd.return_number_coke()
            if start_end_coke is None:
                exit("Координаты зон существуют, а номера нет")

            numbers_coke = [x for x in range(start_end_coke[0], start_end_coke[1] + 1)]
            side_coke = start_end_coke[2]
            if len(numbers_coke) != len(coord_coke):
                exit("Количество зон не совпадает с количеством номеров")

            for i in range(len(coord_coke)):
                self.coordinates_roi.append(coord_coke[str(i)])
                self.fake_coordinates_roi.append(coord_coke_fake[str(i)])
                self.numbers.append(numbers_coke[i])
                self.side.append(side_coke)

        if coord_machine is not None and coord_machine_fake is not None:
            start_end_machine = jd.return_number_machine()
            if start_end_machine is None:
                exit("Координаты зон существуют, а номера нет")

            numbers_machine = [x for x in range(start_end_machine[0], start_end_machine[1] + 1)]
            side_machine = start_end_machine[2]
            if len(numbers_machine) != len(coord_machine):
                exit("Количество зон не совпадает с количеством номеров")

            for i in range(len(coord_machine)):
                self.coordinates_roi.append(coord_machine[str(i)])
                self.fake_coordinates_roi.append(coord_machine_fake[str(i)])
                self.numbers.append(numbers_machine[i])
                self.side.append(side_machine)


class DatasetDirectoryController:
    MAIN_DATASET_DIR = "dataset"

    _path_for_labels: List[str]
    _dataset_main_dir: str

    def __init__(self,
                 dataset_main_dir: Optional[str] = None):
        self._path_for_labels = []
        self._dataset_main_dir = dataset_main_dir or self.MAIN_DATASET_DIR

    def get_directory_for_label_idx(self, label_idx: int) -> str:
        return self._path_for_labels[label_idx]

    def prepare_directories(self):
        if not os.path.exists(self._dataset_main_dir):
            os.mkdir(self._dataset_main_dir)
        for label in LABELS:
            label_path = os.path.join(self._dataset_main_dir, label)
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            self._path_for_labels.append(label_path)

@attr.s
class DetectionPlace:
    place: str = attr.ib()
    detection: List = attr.ib()
    score: float = attr.ib()
    label: str = attr.ib()
    label_idx: int = attr.ib()
    line_thickness: int = attr.ib(default=4)


class ImageDetectionController:
    _detections_of_image: Dict[str, List[Dict[str, Any]]]

    def __init__(self):
        self._detections_of_image = {}

    def __len__(self):
        return len(self._detections_of_image.items())

    def add(self,
            img_path: str,
            img_detections: List[DetectionPlace],
            line_thickness: Optional[int] = None):
        if line_thickness:
            for img_det in img_detections:
                img_det.line_thickness = line_thickness

        detections = self._detections_of_image.get(img_path)
        if not detections:
            self._detections_of_image[img_path] = list(map(attr.asdict, img_detections))
        else:
            self._detections_of_image[img_path].extend(list(map(attr.asdict, img_detections)))

    def save(self,
             path_to_save_detections: str):
        with open(path_to_save_detections, 'w') as f:
            json.dump(self._detections_of_image, f, indent=4)


class ImageDrawer:
    # CATEGORY_INDEX = {
    #     1: {'id': 1, 'name': EMISSION_LABEL},
    # }
    CATEGORY_INDEX = {
        label_idx: {"id": label_idx, "name": label}
        for label_idx, label in enumerate(LABELS)
    }

    _detection_controller: ImageDetectionController
    _matrix: List[List]
    _max_width: int
    _max_height: int

    def __init__(self,
                 detection_controller: ImageDetectionController,
                 matrix: List[List],
                 max_width: int,
                 max_height: int):
        self._detection_controller = detection_controller
        self._matrix = matrix
        self._max_width = max_width
        self._max_height = max_height

    def random_image(self):
        return random.choice(list(self._detection_controller._detections_of_image.keys()))

    def draw_image_with_box(self,
                            dir_to_save: str,
                            image_path: str,
                            use_normalized_coordinates: bool):
        if not os.path.exists(dir_to_save):
            os.mkdir(dir_to_save)

        img_data = self._detection_controller._detections_of_image[image_path]
        name = os.path.basename(image_path)
        image_np = np.array(Image.open(image_path))
        image_np = cv2.warpPerspective(
            image_np,
            np.array(self._matrix),
            (self._max_width, self._max_height),
            flags=cv2.INTER_LINEAR
        )
        plot_detections(
            image_np=image_np,
            boxes=np.array([np.array(box["detection"]) for box in img_data]),
            classes=np.array([box["label_idx"] for box in img_data]),
            scores=np.array([box["score"] for box in img_data]),
            category_index=self.CATEGORY_INDEX,
            image_name=os.path.join(dir_to_save, name),
            line_thickness=[box["line_thickness"] for box in img_data],
            use_normalized_coordinates=use_normalized_coordinates
        )

    def draw_random_images(self,
                           number: int,
                           dir_to_save: str,
                           use_normalized_coordinates: bool):
        for _ in range(number):
            rnd_image = self.random_image()
            self.draw_image_with_box(dir_to_save=dir_to_save,
                                     image_path=rnd_image,
                                     use_normalized_coordinates=use_normalized_coordinates)

