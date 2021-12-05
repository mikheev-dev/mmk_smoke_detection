from typing import Optional

import datetime
import numpy as np
import logging
import cv2

from config import Config
from data import QFrame
from tasks import BaseProcessTask, measure_time_and_log, QLogContext


TIME_FORMAT = "%m-%d-%Y_%H:%M:%S"


class CameraReader(BaseProcessTask):
    """
        Класс для непрерывного чтения в процессе
    """
    _conf_key: str
    _stream: cv2.VideoCapture
    _reconnect_counter: int = 0
    _fps: int

    _address: str

    def __init__(
            self,
            context: QLogContext,
            conf_key: str,
            config: Config,
            fps: int = 12
    ) -> None:
        super().__init__(context)
        self._conf_key = conf_key
        self._address = config.camera_address
        self._fps = fps

    def _connect(self) -> None:
        self._logger.info(f"[{self._address}]::_connect: Connecting to the camera [], "
                          f"with buffer size [{cv2.CAP_PROP_BUFFERSIZE}],"
                          f"and fps [{self._fps}]")
        self._stream = cv2.VideoCapture(self._address)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._logger.info(
            f"[{self._address}]::_connect:Connect success to the camera stream {self._address}")

    def _reconnect(self):
        self._stream = cv2.VideoCapture(self._address)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._logger.info(
            f"[{self._address}]::_connect:ReConnect success to the camera stream {self._address}")

    def handle_frame(
            self,
            frame: np.ndarray
    ) -> Optional[np.ndarray]:
        self._logger.debug(
            f"[{self._address}]::save_frame: Putting frame from stream [{self._address}]")
        if not isinstance(frame, np.ndarray):
            self._logger.debug(
                f"[{self._address}]::save_frame: Frame is empty!")
            return None
        return frame

    @measure_time_and_log(msg="CamReader:: Read 1 frame from camera")
    def stream_read(self) -> Optional[np.ndarray]:
        _, frame = self._stream.read()
        return frame

    def _setup(self):
        self._logger.info(f"[{self._address}]::_connect: Starting streaming")
        self._connect()
        super()._setup()

    def _main(self) -> None:
        _, frame = self._stream.read()
        if isinstance(frame, np.ndarray):
            frame = self.handle_frame(frame=frame)
            if not frame:
                return
            self.dst_q.put(
                QFrame(
                    ts=datetime.datetime.now().strftime(TIME_FORMAT),
                    frame=frame,
                    conf_key=self._conf_key
                )
            )
        elif frame is None:
            self._logger.debug(f"[{self._address}]::run:Camera stream is empty")
            self._reconnect()
        else:
            self._logger.warning(f"[{self._address}]::run: Unsupported situation!")
