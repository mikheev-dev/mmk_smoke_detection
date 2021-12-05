from PIL import Image
from typing import List, Optional

import attr
import base64
import cv2
import os
import numpy as np
import uuid

CAMERA_KEYS = [
    "ct131401",
    "ct131402",
    "ct131403",
    "ct131404",
    "ct131405",
    "ct131406"
]

BACKGROUND_LABEL = "background"
EMISSION_LABEL = "emission"
LABELS = [BACKGROUND_LABEL, EMISSION_LABEL]

EXPAND_HEIGHT = 60
EXPAND_WIDTH = 120

DEFAULT_BATCH_SIZE = 128
DEFAULT_TIMEOUT = 10

DEFAULT_SENT_IMG_SIZE = (1280, 720)
RESIZE_DIM = (224, 224)


@attr.s
class Box:
    place: int = attr.ib()
    x_max: int = attr.ib()
    x_min: int = attr.ib()
    y_max: int = attr.ib()
    y_min: int = attr.ib()
    emission_prediction: float = attr.ib(default=0.0, init=False)

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list())


@attr.s
class QSplittedFrameByBox:
    frame: np.ndarray = attr.ib()
    box: Box = attr.ib()
    ts: Optional[str] = attr.ib()

    zone_name: str = attr.ib(init=False)
    cfg_key: str = attr.ib(init=False)
    draw_frame: bool = attr.ib(init=False)

    def draw(self):
        cv2.rectangle(self.frame,
                      (self.box.x_min, self.box.y_min),
                      (self.box.x_max, self.box.y_max),
                      (255, 0, 0),
                      4)
        proc = np.around(float(self.box.emission_prediction), 2) * 100
        cv2.putText(self.frame, f'Emissions:{int(proc)}%',
                    (self.box.x_min - 5, self.box.y_max + 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def encode_image(self) -> Optional[str]:
        """
        Кодирует изображения в строку
        """
        img = cv2.resize(self.frame, DEFAULT_SENT_IMG_SIZE)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img = cv2.imencode('.jpg', img, encode_param)
        if not result:
            return None
        frame = base64.b64encode(img).decode("utf-8")
        return frame


class QFrame:
    frame: np.ndarray
    conf_key: str
    ts: str

    boxes: List[Box]
    pers_frame: np.ndarray

    def __init__(
            self,
            frame: np.ndarray,
            ts: str,
            conf_key: str
    ):
        self.frame = frame
        self.ts = ts
        self.conf_key = conf_key

    def split_by_box(self) -> List[QSplittedFrameByBox]:
        return [
            QSplittedFrameByBox(
                frame=self.frame,
                box=box,
                ts=self.ts
            )
            for box in self.boxes
        ]

    def draw_and_save(
            self,
            path_to_save: str
    ):
        dframe = self.frame
        for box in self.boxes:
            if box.emission_prediction > 0.001:
                cv2.rectangle(
                    dframe,
                    (box.x_min, box.y_min),
                    (box.x_max, box.y_max),
                    (255, 0, 0),
                    4
                )
                proc = np.around(float(box.emission_prediction), 2) * 100
                cv2.putText(dframe, f'Emissions:{int(proc)}%',
                            (box.x_min - 5, box.y_max + 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        Image.fromarray(dframe).save(
            os.path.join(
                path_to_save,
                f"{uuid.uuid4()}.jpg"
            )
        )


class QBatch:
    frames: List[QFrame]
    batch: np.ndarray

    def __init__(
            self,
            frames: List[QFrame],
            batch: np.ndarray
    ):
        self.frames = frames
        self.batch = batch

