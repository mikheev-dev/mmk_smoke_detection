import datetime
from typing import Optional
import hikvisionapi as hv
from multiprocessing import get_logger as m_get_logger
import numpy as np
import threading
import logging
import time
import re
import cv2
import os

DEFAULT_CHUNK_SIZE = 17000
TIME_FORMAT = "%m-%d-%Y_%H:%M:%S"
DEFAULT_FORMAT = ".jpg"

def parse_url(to_parse: str):
    user = re.search(r'//[a-z0-9A-Z]*:', to_parse)[0].replace("//", "").replace(":", "")
    password = re.search(r':[a-z0-9A-Z!]*@', to_parse)[0].replace("@", "").replace(":", "")
    ip = re.search(r'[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}', to_parse)[0]
    return ip, user, password



def get_logger(level=logging.INFO) -> logging.Logger:
    logger = m_get_logger()
    formatter = logging.Formatter("%(asctime)s]:[%(name)s]:{%(levelname)s}:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class CamReader(threading.Thread):
    """
        Класс для непрерывного чтения в потоке
    """
    _stream: cv2.VideoCapture
    _logger: logging.Logger
    _frame: Optional = None
    _stamp: datetime.datetime = 0
    _address: str
    _reconnect_counter: int = 0
    _fps: int
    _save_folder: str
    _ip: str = ""

    def __init__(self,
                 address: str, name: str, logger: logging.Logger,
                 save_folder: str, fps: int = 12) -> None:
        super().__init__(name=name)
        self._logger = logger
        self._address = address
        self._fps = fps
        self._prepare_folder(save_folder)

    def _connect(self) -> None:
        self._logger.info(f"[{self.name}]::_connect: Connecting to the camera [], "
                          f"with buffer size [{cv2.CAP_PROP_BUFFERSIZE}],"
                          f"and fps [{self._fps}]")
        self._stream = cv2.VideoCapture(self._address)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._logger.info(
            f"[{self.name}]::_connect:Connect success to the camera stream {self._address}")

    def _reconnect(self):
        self._stream = cv2.VideoCapture(self._address)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._logger.info(
            f"[{self.name}]::_connect:ReConnect success to the camera stream {self._address}")

    def _prepare_folder(self, path: str):
        ip, user, password = parse_url(self._address)
        self._ip = ip
        target_dir = os.path.join(path, ip)
        try:
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            self._save_folder = target_dir
        except FileExistsError:
            self._save_folder = os.path.join(os.getcwd(), ip)
            os.mkdir(self._save_folder)
        self._logger.info(
            f"[{self.name}]::_prepare_folder: Save folder for cam [{ip}] > [{self._save_folder}]")

    def save_frame(self):
        self._logger.debug(
            f"[{self.name}]::save_frame: Saving frame from stream [{self._address}]")
        if not isinstance(self._frame, np.ndarray):
            self._logger.debug(
                f"[{self.name}]::save_frame: Frame is empty!")
            return
        file_name = f"frame_{self._stamp.strftime(TIME_FORMAT)}.jpg"
        filepath = os.path.join(self._save_folder, file_name)
        cv2.imwrite(os.path.abspath(filepath), self._frame)
        self._logger.debug(f"[{self.name}]::save_frame: Frame saved by path [{filepath}]]")

    def show_frame(self):
        if not isinstance(self._frame, np.ndarray):
            self._logger.debug(
                f"[{self.name}]::show_frame: Frame is empty!")
            return
        cv2.imshow(f"{self.name}", self._frame)
        cv2.waitKey(1)

    def run(self) -> None:
        self._logger.info(f"[{self.name}]::_connect: Starting streaming")
        self._connect()
        while True:
            _, frame = self._stream.read()
            if isinstance(frame, np.ndarray):
                self._frame = frame
                self._stamp = datetime.datetime.now()
            elif frame is None:
                self._logger.debug(f"[{self.name}]::run:Camera stream is empty")
                self._reconnect()
            else:
                self._logger.warning(f"[{self.name}]::run: Unsupported situation!")

    def get_frame(self):
        return self._frame, self._stamp


if __name__ == '__main__':
    log = get_logger(logging.INFO)
    cameras = [
        "rtsp://root:Video5000@172.30.71.99/axis-media/media.amp?",
        "rtsp://root:Video5000@172.30.71.100/axis-media/media.amp?",
        "rtsp://root:Video5000@172.30.71.101/axis-media/media.amp?",
        "rtsp://root:Video5000@172.30.71.102/axis-media/media.amp?",
        "rtsp://root:Video5000@172.30.71.103/axis-media/media.amp?",
        "rtsp://root:Video5000@172.30.71.104/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.105/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.106/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.107/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.108/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.109/axis-media/media.amp?",
        "rtsp://root:Video5000!@172.30.71.110/axis-media/media.amp?"
    ]

    SAVE_FOLDER = "./flow_data"
    FPS = 0.2
    log.info(f"Creating Threads")
    recs = [CamReader(
        name=f"Cam_writer_{(itt+100)-1}",
        logger=log,
        address=cam,
        save_folder=SAVE_FOLDER
    ) for itt, cam in enumerate(cameras)]
    log.info(f"Creating have been created!")
    log.info(f"Starting Threads")
    for rec in recs:
        rec.start()
    log.info(f"Threads starter")
    start_time = time.time()
    log.info(f"Starting checkout loop")

    while True:
        # log.info(f"Parsing Cams")
        for cam in recs:
            cam.save_frame()
            cam.show_frame()
        # log.info(f"Cams parsing finished...")
        diff = (start_time + (1 / FPS))
        if diff > time.time():
            time.sleep(diff - time.time())
        start_time = time.time()
