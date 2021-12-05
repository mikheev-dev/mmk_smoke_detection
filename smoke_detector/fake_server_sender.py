import json
import os.path

from data import QSplittedFrameByBox
from PIL import Image
from smoke_detector.server_sender import RoiServerSender
from tasks import measure_time_and_log, QLogContext
from config_controller import ConfigController


class FakeRoiServerSender(RoiServerSender):
    _result_path: str

    def __init__(
            self,
            result_path: str,
            context: QLogContext,
            cfg_controller: ConfigController,
    ):
        super().__init__(
            context=context,
            cfg_controller=cfg_controller
        )
        self._result_path = result_path

    @measure_time_and_log(msg="FakeRoiServerSender:: Save 1 frame")
    def _handle_frame(
            self,
            frame_box: QSplittedFrameByBox
    ) -> None:
        cfg = self._cfg_controller[frame_box.cfg_key]
        data = self._prepare_data(cfg, frame_box)
        data = json.loads(data)
        data = data["data"]
        number = data["number"]
        name = f"{frame_box.ts}.jpg"
        print(os.path.join(self._result_path, name))
        # Image.fromarray(frame_box.frame).save(
        #     os.path.join(self._result_path, number, name)
        # )
