from typing import Optional
import hikvisionapi as hv
from multiprocessing import get_logger as m_get_logger
import numpy as np
import threading
import logging
import time
import re
import cv2

DEFAULT_CHUNK_SIZE = 17000

def parse_url(to_parse: str):
    user = re.search(r'//[a-z0-9A-Z]*:', to_parse)[0].replace("//", "").replace(":", "")
    password = re.search(r':[a-z0-9A-Z]*@', to_parse)[0].replace("@", "").replace(":", "")
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


class HyperCamReader(threading.Thread):
    _reconnect_counter = 0
    _camera: hv.hikvisionapi.Client
    _logger: logging.Logger
    _frame: Optional = None
    _stamp: float = 0
    _address: str
    _fps: int

    def __init__(self, address: str, name: str, logger: logging.Logger,):
        super().__init__(name=name)
        self._logger = logger
        self._address = address

    def connect(self):
        ip, user, password = parse_url(self._address)
        self._logger.info(
            f"[{self.name}]::_connect connecting to the IP [http://{ip}], USER [{user}],PSWD [{password}]")
        self._camera = hv.Client(f'http://{ip}', user, password)

        self._logger.info(
            f"[{self.name}]::_connect:Connect success to the camera stream {self._address} "
            f"and device {self._camera.System.deviceInfo(method='get')}")

    def run(self) -> None:
        self.connect()
        while True:
            stream = self._camera.Streaming.channels[101].picture(method='get', type='opaque_data')
            body = b''
            for chunk in stream.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                body += chunk
                start = body.find(b'\xff\xd8')
                end = body.find(b'\xff\xd9')
                if start != -1 and end != -1:
                    img = body[start:end + 2]
                    body = body[end + 2:]
                    self._frame = cv2.imdecode(np.fromstring(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self._stamp = time.time()

    def get_frame(self):
        return self._frame, self._stamp

