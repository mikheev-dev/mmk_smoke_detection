from typing import Dict

import datetime

from config import Config, RoiLoader
from config_controller import ConfigController
from data import QFrame, QSplittedFrameByBox
from logging import Logger
from multiprocessing import Queue

from tasks import BaseProcessTask, QLogContext
from typing import List


TIME_BORDER = 60


class RoiTrigger:
    """
    One trigger for one camera
    """
    _global_counter: int
    _counter: int
    _last_time: datetime.datetime

    _global_counter_border: int
    _counter_border: int

    _is_active: bool = False
    _every_third_flag: int = 0

    def __init__(
            self,
            cfg: Config
    ):
        self._last_time = datetime.datetime.now()
        self._cfg = cfg
        self._counter_border = cfg.count
        self._global_counter_border = cfg.global_count
        self._zero_counters()

    def _zero_counters(self) -> None:
        self._counter = 0
        self._global_counter = 0

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def need_sent_frame(self) -> bool:
        return not self._every_third_flag % 3

    def deactivate(self):
        self._is_active = False
        self._zero_counters()

    def _check_time(self) -> bool:
        current_time = datetime.datetime.now()
        need_to_sent = False
        if self._counter or self._global_counter:
            if (current_time - self._last_time).total_seconds() > TIME_BORDER:
                self._zero_counters()
                need_to_sent = True
        self._last_time = current_time
        return need_to_sent

    def _check_trigger(self):
        need_to_sent = self._check_time()
        if need_to_sent:
            self._every_third_flag = 0
            self._is_active = True
            return
        if self._global_counter >= self._global_counter_border:
            self._every_third_flag = (self._every_third_flag + 1) % 3
            self._is_active = True
            return

    def update_counters(self) -> None:
        self._counter += 1
        if self._counter >= self._counter_border:
            self._global_counter += 1
            self._counter = 0
        self._check_trigger()


class CameraTriggerController(BaseProcessTask):
    _cfg_key: str
    _cfg: Config
    _zone_name: str

    _triggers: List[RoiTrigger]

    def __init__(
            self,
            context: QLogContext,
            cfg_key: str,
            cfg: Config,
            zone_name: str
    ):

        super().__init__(context)
        self._cfg_key = cfg_key
        self._cfg = cfg
        self._zone_name = zone_name
        roi = RoiLoader(cfg=cfg)
        self._triggers = [
            RoiTrigger(cfg=cfg)
            for _ in range(len(roi.coordinates_roi))
        ]

    def _handle_frame(
            self,
            qframe: QFrame
    ):
        splitted_frame: List[QSplittedFrameByBox] = qframe.split_by_box()
        for roi_idx, frame_box in enumerate(splitted_frame):
            trigger = self._triggers[roi_idx]
            trigger.update_counters()
            if trigger.is_active:
                frame_box.draw_frame = trigger.need_sent_frame
                frame_box.cfg_key = self._cfg_key
                frame_box.zone_name = self._zone_name

                self.dst_q.put(frame_box)
                trigger.deactivate()

    def _main(self):
        frame = self.src_q.get()
        self._handle_frame(frame)


class CameraTriggerMapper(BaseProcessTask):
    _cfg_controller: ConfigController
    _camera_senders: Dict[str, CameraTriggerController] = dict()
    _internal_qs: Dict[str, Queue] = dict()

    def __init__(
            self,
            context: QLogContext,
            cfg_controller: ConfigController
    ):
        super().__init__(context=context)
        self._cfg_controller = cfg_controller
        for cfg_key, cfg, zone_name in iter(cfg_controller):
            self._internal_qs[cfg_key] = Queue()
            
    def _init_camera_senders(self):
        for cfg_key, cfg, zone_name in iter(self._cfg_controller):
            context = QLogContext()
            context.src_q = self._internal_qs[cfg_key]
            context.dst_q = self.dst_q
            context.logger = self.logger
            camera_sender = CameraTriggerController(
                context=context,
                cfg_key=cfg_key,
                cfg=cfg,
                zone_name=zone_name,
            )
            self._camera_senders[cfg_key] = camera_sender

    def _setup(self):
        self._init_camera_senders()
        super()._setup()
        for camera_sender in self._camera_senders.values():
            camera_sender.start()

    def _main(self):
        frame: QFrame = self.src_q.get()
        self._internal_qs[frame.conf_key].put(frame)
