"""
Принимает сохраненный JSON файлики в моем формате.
Дальше берет ключи (изображения) и бежит по всем по ним.

Принимает config JsonData.

Изображение кропится на плейсы. По каждой детекции идет поиск ближайшего плейса (по IoU ???)
Строим отношение плейс -> боксы с метками

По каждому плейсу делаем голосование
Как понять, что голосуем?

Давай отображу все на одном файлике

"""
from typing import List, Dict, Set, Any, Tuple, Optional
from preparator import ImageDetectionController
from preparator import DetectionPlace
from preparator import ImageDrawer
from classJson import JsonData

import json


class ImageDetectionsMerger:
    _models_detections: List[Dict[str, List[DetectionPlace]]]
    _images: Set[str]
    _merged_images_detections_controller: ImageDetectionController

    def __init__(self):
        self._models_detections = []
        self._images = set()
        self._merged_images_detections_controller = ImageDetectionController()

    @property
    def detection_controller(self) -> ImageDetectionController:
        return self._merged_images_detections_controller

    @staticmethod
    def _replace_image_name(image_name: str,
                            replacing_patterns: Optional[List[Tuple[str, str]]]) -> str:
        if replacing_patterns:
            new_image_name = image_name
            for pattern in replacing_patterns:
                new_image_name = new_image_name.replace(pattern[0], pattern[1])
            return new_image_name
        else:
            return image_name

    def merge(self,
              path_to_vanilla_detections: str,
              pathes_to_models_detections: List[str],
              path_to_merged_json: str,
              replacing_patterns: Optional[List[Tuple[str, str]]] = None):
        pathes = [path_to_vanilla_detections] + pathes_to_models_detections
        # First detections is for vanilla models!
        self._models_detections = []
        for path in pathes:
            with open(path) as p:
                self._models_detections.append({
                    self._replace_image_name(image, replacing_patterns): [
                        DetectionPlace(**place)
                        for place in places
                    ]
                    for image, places in json.load(p).items()
                })
        self._images = set()
        for detections in self._models_detections:
            self._images |= set(detections.keys())

        for image in self._images:
            for model_idx, model_detections in enumerate(self._models_detections):
                image_model_detections = model_detections.get(image)
                if image_model_detections:
                    self._merged_images_detections_controller.add(
                        img_path=image,
                        img_detections=image_model_detections,
                        line_thickness=3 + 2 * model_idx
                    )

        self._merged_images_detections_controller.save(
            path_to_save_detections=path_to_merged_json
        )


if __name__ == "__main__":
    PATH_TO_VANILLA_DETECTIONS = "vanilla_test_images/vanilla_detections.json"
    PATH_TO_SMOKE_DETECTIONS = "smoke_test_images/smoke_detections.json"
    PATH_TO_FIRE_DETECTIONS = "fire_test_images/fire_detections.json"
    CONFIG_PATH = "./conf/conf_ct131405.json"

    conf = JsonData(name=CONFIG_PATH)
    data_matrix = conf.return_matrix()
    matrix = data_matrix["matrix"]
    max_height = data_matrix["maxHeight"]
    max_width = data_matrix["maxWidth"]

    merger = ImageDetectionsMerger()
    merger.merge(
        path_to_merged_json="./merged_detections.json",
        path_to_vanilla_detections=PATH_TO_VANILLA_DETECTIONS,
        pathes_to_models_detections=[
            PATH_TO_SMOKE_DETECTIONS,
            PATH_TO_FIRE_DETECTIONS
        ],
        replacing_patterns=[
            ("/tf", "/home/odyssey"),
            ("/app", "/home/odyssey")
        ]
    )
    # print(merger.detection_controller.itemsems())
    drawer = ImageDrawer(
        detection_controller=merger.detection_controller,
        max_height=max_height,
        max_width=max_width,
        matrix=matrix
    )
    MERGED_IMAGES = "./merged_test_images"
    drawer.draw_random_images(number=100,
                              dir_to_save=MERGED_IMAGES,
                              use_normalized_coordinates=False)
