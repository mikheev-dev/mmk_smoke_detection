from multiprocessing import get_logger as get_mp_logger
from multiprocessing import Queue
from typing import Callable, Any, Optional, List, Union, Iterator

import logging
import os
import shutil

from camera_reader import CameraReader
from config_controller import ConfigController
from preprocess import Preprocess
from get_env import get_env
from inference import Inference
from camera_mapper import CameraTriggerMapper
from server_sender import RoiServerSender
from tasks import BaseProcessTask, TaskController, QLogContext

from fake_camera_reader import FakeCameraReader
from fake_server_sender import FakeRoiServerSender


def get_logger(log_level) -> logging.Logger:
    level = log_level
    logger = get_mp_logger()
    formatter = logging.Formatter("%(asctime)s]:[%(name)s]:{%(levelname)s}:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


LOG_LEVEL = get_env("LOG_LEVEL", default="INFO")
MULTIPROCESS_LOGGER = get_logger(LOG_LEVEL)
ZONE_NAMES = get_env("ZONE_NAMES", cast=lambda x: x.strip().split(','))
PATH_TO_CONFIGS = get_env("PATH_TO_CONFIGS", cast=str)
CONFIG_FILE_NAMES = get_env("CONFIG_FILE_NAMES", cast=lambda x: x.strip().split(','))
CAMERA_FPS = get_env("CAMERA_FPS", cast=int)
PATH_TO_MODEL = get_env("PATH_TO_MODEL")
BATCH_SIZE = get_env("BATCH_SIZE", cast=int)
INFERENCE_GET_TIMEOUT = get_env("INFERENCE_GET_TIMEOUT", cast=float)
SERVER_SENDERS_COUNT = get_env("SERVER_SENDERS_COUNT", cast=int, default=5)

IS_TESTING = get_env("IS_TESTING", cast=bool, default=False)
PATH_TO_IMAGES = get_env("PATH_TO_IMAGES", cast=lambda x: x.strip().split(',')) 
RESULT_PATH = get_env("RESULT_PATH")

NUMBER_OF_LINKS = 6


def create_links() -> Iterator[QLogContext]:
    """
    Q -> camera_reader -> Q -> preprocess -> Q -> inference -> Q -> camera_mapper -> Q -> server_sender -> Q
    => 6 links 
    """
    links = []
    qs = [Queue() for _ in range(NUMBER_OF_LINKS)]
    for src_q, dst_q in zip(qs[:-1], qs[1:]):
        link = QLogContext()
        link.src_q = src_q
        link.dst_q = dst_q
        link.logger = MULTIPROCESS_LOGGER
        links.append(link)
    return iter(links)


def init_camera_readers(
        context: QLogContext,
        cfg_controller: ConfigController,
        need_fake: bool
) -> List[Union[CameraReader, FakeCameraReader]]:
    if not need_fake:
        return [
            CameraReader(
                conf_key=cfg_key,
                config=cfg,
                fps=CAMERA_FPS,
                context=context
            )
            for cfg_key, cfg, _ in iter(cfg_controller)
        ]
    if not PATH_TO_IMAGES:
        msg_err = "Need pathes to images for testing pipeline!"
        MULTIPROCESS_LOGGER.error(msg_err)
        raise Exception(msg_err)

    return [
        FakeCameraReader(
            conf_key=cfg_key,
            path_to_images=path_to_imgs,
            fps=CAMERA_FPS,
            context=context
        )
        for (cfg_key, cfg, _), path_to_imgs in zip(iter(cfg_controller), PATH_TO_IMAGES)
    ]


def init_server_senders(
        context: QLogContext,
        cfg_controller: ConfigController,
        need_fake: bool
) -> List[Union[RoiServerSender, FakeRoiServerSender]]:
    if need_fake:
        if not RESULT_PATH:
            err_msg = "No RESULT_PATH for test mode!"
            MULTIPROCESS_LOGGER.error(err_msg)
            raise Exception(err_msg)
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.makedirs(RESULT_PATH)
        return [
            FakeRoiServerSender(
                cfg_controller=cfg_controller,
                context=context,
                result_path=RESULT_PATH
            )
            for _ in range(SERVER_SENDERS_COUNT)
        ]

    return [
        RoiServerSender(
            cfg_controller=cfg_controller,
            context=context
        )
        for _ in range(SERVER_SENDERS_COUNT)
    ]


def prepare_pipeline() -> TaskController:
    cfg_controller = ConfigController(
        path_to_configs=PATH_TO_CONFIGS,
        config_file_pathes=CONFIG_FILE_NAMES
    )
    cfg_controller.zone_names = ZONE_NAMES
    
    links: Iterator[QLogContext] = create_links() 

    tasks = []

    # =====  Camera Reader =====
    camera_readers = init_camera_readers(
        cfg_controller=cfg_controller,
        need_fake=IS_TESTING,
        context=next(links)
    )
    tasks.extend(camera_readers)

    # =====  Preprocess =====
    expander = Preprocess(
        config_controller=cfg_controller,
        context=next(links),
        batch_size=BATCH_SIZE,
        timeout=INFERENCE_GET_TIMEOUT,
    )
    tasks.append(expander)

    # =====  Inference =====
    inference = Inference(
        path_to_model=PATH_TO_MODEL,
        context=next(links)
    )
    tasks.append(inference)

    # =====  Camera Mapper =====
    camera_trigger_mapper = CameraTriggerMapper(
        cfg_controller=cfg_controller,
        context=next(links)
    )
    tasks.append(camera_trigger_mapper)

    # ===== Server Senders =====
    server_senders = init_server_senders(
        cfg_controller=cfg_controller,
        need_fake=IS_TESTING,
        context=next(links)
    )
    tasks.extend(server_senders)

    return TaskController(
        tasks=tasks,
        logger=MULTIPROCESS_LOGGER
    )


if __name__ == "__main__":
    shutil.rmtree("test_inference")
    os.makedirs("test_inference")
    task_controller: TaskController = prepare_pipeline()
    task_controller.start()
    task_controller.join()
