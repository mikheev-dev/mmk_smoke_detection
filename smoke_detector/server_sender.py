import json
from logging import Logger

import requests

from smoke_detector.config import Config
from smoke_detector.config_controller import ConfigController
from smoke_detector.data import QSplittedFrameByBox
from smoke_detector.tasks import BaseProcessTask, measure_time_and_log, QLogContext


class RoiServerSender(BaseProcessTask):
    _cfg_controller: ConfigController

    def __init__(
            self,
            context: QLogContext,
            cfg_controller: ConfigController,
    ):
        super().__init__(context)
        self._cfg_controller = cfg_controller

    @staticmethod
    def _prepare_data(
            cfg: Config,
            frame_box: QSplittedFrameByBox
    ) -> str:
        """
        Создает необходимый формат для отправки на сервер
         number, side, x_max, y_max, x_min, y_min, predict, image=None, zone_name=""
        """
        real_number = cfg.min_number_coke + int(frame_box.box.place)
        data = {
            "name": cfg.side_coke,
            "number": real_number,
            "zone_name": frame_box.zone_name
        }
        if frame_box.draw_frame:
            frame_box.draw()
            encoded_img = frame_box.encode_image()
            if encoded_img:
                data.update({
                    "image": encoded_img
                })
        return json.dumps({"data": data})

    @measure_time_and_log(msg="Send 1 frame to server")
    def _handle_frame(
            self,
            frame_box: QSplittedFrameByBox
    ) -> None:
        cfg = self._cfg_controller[frame_box.cfg_key]
        data = self._prepare_data(cfg, frame_box)
        address_to_server = f"http://{cfg.server_address}:{cfg.server_port}"
        headers = {'Content-type': 'application/json'}
        request = requests.post(address_to_server, data=data, headers=headers)
        print(request.status_code)

    def _main(self) -> None:
        frame_box: QSplittedFrameByBox = self.src_q.get()
        self._handle_frame(frame_box)
