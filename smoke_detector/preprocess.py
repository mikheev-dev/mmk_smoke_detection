from typing import List, Tuple, Dict

import cv2
import numpy as np

from data import Box, QFrame, QBatch
from data import EXPAND_WIDTH, EXPAND_HEIGHT, RESIZE_DIM, DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT

from tasks import BaseProcessTask, measure_time_and_log, QLogContext
from config import CameraConfig
from config_controller import ConfigController

from PIL import Image
import uuid


class Preprocess(BaseProcessTask):
    _cfg_controller: ConfigController
    _expanded_boxes_for_cameras: Dict[str, List[Box]] = dict()
    _boxes_for_cameras: Dict[str, List[Box]] = dict()

    _batch_size: int
    _get_timeout: int

    _batch: List[np.ndarray]
    _frames: List[QFrame]

    def __init__(
            self,
            context: QLogContext,
            config_controller: ConfigController,
            batch_size: int = DEFAULT_BATCH_SIZE,
            timeout: int = DEFAULT_TIMEOUT
    ):
        super().__init__(context)
        self._cfg_controller = config_controller
        print(list(config_controller.camera_configs.keys()))
        self._prepare_boxes()
        self._prepare_expanded_boxes()
        self._get_timeout = timeout
        self._batch_size = batch_size
        self._flush()

    def _flush(self) -> None:
        self._frames = []
        self._batch = []

    def _prepare_boxes(self):
        for cfg_name, camera_config in self._cfg_controller.camera_configs.items():
            self._boxes_for_cameras[cfg_name] = [
                Box(
                    place=idx,
                    x_max=region["x_max"],
                    x_min=region["x_min"],
                    y_max=region["y_max"],
                    y_min=region["y_min"],
                )
                for idx, region in enumerate(camera_config.roi.coordinates_roi)
            ]

    def _apply_perspective_to_boxes(
            self,
            camera_config: CameraConfig,
            boxes: List[Box]
    ) -> List[Box]:
        np_boxes = np.array([box.to_numpy() for box in boxes])
        reshape_boxes = np.float32(np_boxes.reshape(-1)).reshape(-1, 1, 2)
        camera_matrix = camera_config.matrix
        pers_boxes = cv2.perspectiveTransform(reshape_boxes, camera_matrix).reshape(-1).astype(int).reshape(-1, 4)
        return [
            Box(
                place=box.place,
                x_min=pers_box[0],
                y_min=pers_box[1],
                x_max=pers_box[2],
                y_max=pers_box[3]
            )
            for pers_box, box in zip(pers_boxes, boxes)
        ]

    def _prepare_expanded_boxes(self):
        for cfg_name, camera_config in self._cfg_controller.camera_configs.items():
            boxes = self._boxes_for_cameras[cfg_name]
            pers_boxes = self._apply_perspective_to_boxes(
                camera_config=camera_config,
                boxes=boxes
            )
            # pers_boxes = boxes
            self._expanded_boxes_for_cameras[cfg_name] = [
                self._expand_box(
                    camera_config=camera_config,
                    box=box
                )
                for box in pers_boxes
            ]

    @staticmethod
    def _increase_box_params(
            cor1: int,
            cor2: int,
            max_value: int,
            coef: float
    ) -> Tuple[int, int]:
        diff = np.abs(cor1 - cor2)
        min_cor = max(0, cor1 - int(coef * diff))
        max_cor = min(max_value, cor2 + int(coef * diff))
        return min_cor, max_cor

    @staticmethod
    def _increase_dim_params(
            c_less: int,
            c_grt: int,
            dim: int,
            max_value: int
    ) -> Tuple[int, int]:
        add_diff = (dim - (c_grt - c_less)) // 2
        less_diff = c_less - add_diff
        grt_diff = max_value - (c_grt + add_diff)

        if less_diff < 0:
            inc_c_less = 0
            inc_c_grt = c_grt + add_diff + np.abs(less_diff)
        elif grt_diff < 0:
            inc_c_grt = max_value
            inc_c_less = c_less - add_diff - np.abs(grt_diff)
        else:
            inc_c_less = c_less - add_diff
            inc_c_grt = c_grt + add_diff

        if inc_c_grt - inc_c_less < dim:
            if inc_c_grt + 1 >= max_value:
                inc_c_less -= 1
            else:
                inc_c_grt += 1

        return inc_c_less, inc_c_grt

    def _expand_box(
            self,
            camera_config: CameraConfig,
            box: Box
    ) -> Box:
        new_min_y, new_max_y = Preprocess._increase_dim_params(box.y_min,
                                                               box.y_max,
                                                               dim=EXPAND_HEIGHT,
                                                               max_value=camera_config.max_height)
        new_min_x, new_max_x = Preprocess._increase_dim_params(box.x_min,
                                                               box.x_max,
                                                               dim=EXPAND_WIDTH,
                                                               max_value=camera_config.max_width)
        box.y_min, box.x_min, box.y_max, box.x_max = new_min_y, new_min_x, new_max_y, new_max_x
        return box

    def _apply_perspective_to_frame(
            self,
            frame: QFrame
    ) -> np.ndarray:
        camera_config = self._cfg_controller[frame.conf_key]
        return cv2.warpPerspective(
            frame.frame,
            camera_config.matrix,
            (camera_config.max_width, camera_config.max_height),
            flags=cv2.INTER_LINEAR
        )

    def _prepare_batch_part(
            self,
            qframe: QFrame
    ) -> np.ndarray:
        pers_frame = self._apply_perspective_to_frame(frame=qframe)
        pers_boxes = self._expanded_boxes_for_cameras[qframe.conf_key]
        qframe.pers_frame = pers_frame
        # qframe.frame = pers_frame
        # qf = QFrame(
        #     frame=pers_frame,
        #     ts=qframe.ts,
        #     conf_key=qframe.conf_key
        # )
        # qf.boxes = pers_boxes
        # qframe.draw_and_save(path_to_save="test_preprocess")

        sub_frames = [
            cv2.resize(
                src=pers_frame[box.y_min: box.y_max, box.x_min: box.x_max],
                dsize=RESIZE_DIM
            )
            for box in pers_boxes
        ]

        # tq = QFrame(
        #     frame=pers_frame,
        #     ts='',
        #     conf_key=''
        # )
        # tq.boxes = pers_boxes
        # stq = tq.split_by_box()
        # prev_frame = None
        # for fb in stq:
        #     if prev_frame is not None:
        #         fb.frame = prev_frame
        #     fb.box.emission_prediction = 0.0
        #     fb.draw()
        #     prev_frame = fb.frame
        # Image.fromarray(prev_frame).save(f"test_result/{uuid.uuid4()}.jpg")

        return np.array(sub_frames)

    def _prepare_batch(self) -> None:
        current_batch_size = 0
        while current_batch_size < self._batch_size:
            try:
                qframe: QFrame = self._src_q.get(timeout=self._get_timeout)
            except Exception:
                if self._batch:
                    return
                continue
            self.logger.debug(f"{self.__class__.__name__}: Get 1 image from q")
            qframe.boxes = self._boxes_for_cameras[qframe.conf_key]
            self._batch.extend(self._prepare_batch_part(qframe))
            self._frames.append(qframe)
            current_batch_size += len(qframe.boxes)

    def _push(self):
        self.dst_q.put(
            QBatch(
                frames=self._frames,
                batch=np.array(self._batch)
            )
        )
        self.logger.info(f"{self.__class__.__name__}: Put 1 batch to q")
        self._flush()

    @measure_time_and_log(msg="Preprocess:: Preparing batch")
    def _main(self):
        self._prepare_batch()
        self._push()
