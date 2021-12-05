from PIL import Image
from typing import List, Iterator

import datetime
import glob
import numpy as np
import os
import time

from data import QFrame
from tasks import BaseProcessTask, QLogContext

TIME_FORMAT = "%m-%d-%Y_%H:%M:%S"


class FakeCameraReader(BaseProcessTask):
    _images: Iterator
    _conf_key: str
    _sleep_time: float

    def __init__(
            self,
            context: QLogContext,
            path_to_images: str,
            conf_key: str,
            fps: int = 12
    ):
        super().__init__(context)
        imgs = glob.glob(os.path.join(path_to_images, "**", "*.jpg"))
        self._images = iter(imgs)
        self._conf_key = conf_key
        self._sleep_time = 1 / fps

    def _main(self):
        try:
            img_path = next(self._images)
        except StopIteration:
            exit()
        img = Image.open(img_path)
        self.logger.info(f"{self.__class__.__name__}: Put 1 image to q")
        self.dst_q.put(
            QFrame(
                ts=datetime.datetime.now().strftime(TIME_FORMAT),
                frame=np.array(img),
                conf_key=self._conf_key
            )
        )
        # to imitate fps
        time.sleep(self._sleep_time)
