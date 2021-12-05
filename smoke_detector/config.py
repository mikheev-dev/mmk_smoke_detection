from typing import List, Tuple, Dict

import json
import os
import numpy as np
import re


class Config:
    """
    Класс, позвляющий работать с Json файлами
    """
    name: str
    json: Dict

    def __init__(self, name):
        """
        :param name: имя файла (обязательно указывать формат)
        """
        self.name = name
        filename, file_extension = os.path.splitext(self.name)
        if str(file_extension) != ".json":
            exit("Необходимо указать расширение файла .json")

        if not os.path.exists(self.name):
            self.json = self.__create_json()
        else:
            self.json = self.__load_json()

    @staticmethod
    def __create_json():
        """
        Создает дефолтный Json файл
        :return:
        """
        data = {"camera_address": None,
                "count": 4,
                "global_count": 5,
                "side": "M",
                "start": 701,
                "end": 765,
                "server_address": "0.0.0.0",
                "server_port": 0000,
                "model_path": "model/leaks.h5",
                "matrix": None,
                "roi": {
                    "roi": None},
                }
        # with open(f"{self.name}", "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        return data

    def save_json(self):
        """
        Сохраняет Json файл, внося в него измнения, если они были
        :return:
        """
        with open(f"{self.name}", "w") as f:
            f.write(json.dumps(self.json))

    def __load_json(self):
        with open(f"{self.name}", "r") as f:
            json_data = json.load(f)
            return json_data

    @property
    def roi(self):
        return self.json.get("roi")

    @roi.setter
    def roi(self, roi):
        rois = {
            idx: {"x_min": r[0][0], "y_min": r[0][1], "x_max": r[1][0], "y_max": r[1][1]}
            for idx, r in enumerate(roi)
        }
        self.json["roi"] = rois
        self.roi_ovens = rois

    @property
    def fake_roi(self):
        return self.json.get("fake_roi")

    @fake_roi.setter
    def fake_roi(self, roi):
        rois = {
            idx: {"x_min": r[0][0], "y_min": r[0][1], "x_max": r[1][0], "y_max": r[1][1]}
            for idx, r in enumerate(roi)
        }
        self.json["fake_roi"] = rois

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.json["matrix"]["matrix"])

    @matrix.setter
    def matrix(self, matrix):
        if not self.json.get("matrix"):
            self.json["matrix"] = {}
        self.json["matrix"]["matrix"] = matrix

    @property
    def max_width(self) -> int:
        return int(self.json["matrix"]["maxWidth"])

    @max_width.setter
    def max_width(self, max_width):
        if not self.json.get("matrix"):
            self.json["matrix"] = {}
        self.json["matrix"]["maxWidth"] = max_width

    @property
    def max_height(self) -> int:
        return int(self.json["matrix"]["maxHeight"])

    @max_height.setter
    def max_height(self, max_height):
        if not self.json.get("matrix"):
            self.json["matrix"] = {}
        self.json["matrix"]["maxHeight"] = max_height

    @property
    def count(self) -> int:
        return self.json["count"]

    @count.setter
    def count(self, count: int):
        self.json["count"] = count

    @property
    def global_count(self) -> int:
        return self.json["global_count"]

    @global_count.setter
    def global_count(self, global_count: int):
        self.json["global_count"] = global_count
        
    @property
    def server_address(self) -> str:
        return self.json["server_address"]

    @property
    def server_port(self) -> str:
        return self.json["server_port"]

    @property
    def camera_address(self) -> str:
        return self.json["camera_address"]

    @property
    def camera_ip(self) -> str:
        camera_url = self.camera_address
        ip = re.search(r"[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}", camera_url)[0]
        return ip

    @property
    def camera_user(self) -> str:
        camera_url = self.camera_address
        user = re.search(r"//[a-z0-9A-Z]*:", camera_url)[0].replace("//", "").replace(":", "")
        return user

    @property
    def camera_password(self) -> str:
        camera_url = self.camera_address
        password = re.search(r":[a-z0-9A-Z!]*@", camera_url)[0].replace("@", "").replace(":", "")
        return password

    @property
    def number_coke(self) -> Tuple[int, int, str]:
        return tuple(self.json["number_coke"])

    @property
    def min_number_coke(self) -> int:
        return self.number_coke[0]

    @property
    def max_number_coke(self) -> int:
        return self.number_coke[1]

    @property
    def side_coke(self) -> str:
        return self.number_coke[2]

    def return_side(self):
        return self.json["side"]

    def return_start_end(self):
        return self.json["start"], self.json["end"]

    def server_data(self):
        return self.json["server_address"], self.json["server_port"]

    def return_model(self):
        return self.json["model_path"]

    def return_roi_machine(self):
        return self.json["roi_machine"]

    def return_roi_coke(self):
        return self.json["roi_coke"]

    def return_number_machine(self):
        return self.json["number_machine"]

    def return_number_coke(self):
        return self.json["number_coke"]

    def return_fake_machine(self):
        return self.json["fake_machine"]

    def return_fake_coke(self):
        return self.json["fake_coke"]


class RoiLoader:
    coordinates_roi: List
    fake_coordinates_roi: List
    numbers: List
    side: List

    def __init__(
            self,
            cfg: Config
    ):
        cfg_roi = cfg.roi
        if cfg_roi:
            self.coordinates_roi = [
                cfg_roi[str(idx)] for idx in range(len(cfg_roi))
            ]
            return

        coord_machine = cfg.return_roi_machine()
        coord_coke = cfg.return_roi_coke()
        coord_machine_fake = cfg.return_fake_machine()
        coord_coke_fake = cfg.return_fake_coke()

        self.coordinates_roi = []
        self.fake_coordinates_roi = []
        self.numbers = []
        self.side = []
        # загрузка координат ROI
        if coord_coke is not None and coord_coke_fake is not None:
            start_end_coke = cfg.return_number_coke()
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
            start_end_machine = cfg.return_number_machine()
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


class CameraConfig:
    matrix: np.ndarray
    max_height: int
    max_width: int
    roi: RoiLoader

    def __init__(
            self,
            config: Config
    ):
        self.matrix = config.matrix
        self.max_width = config.max_width
        self.max_height = config.max_height
        self.roi = RoiLoader(cfg=config)
