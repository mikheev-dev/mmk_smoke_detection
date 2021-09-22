from multiprocessing import Queue
from logging import Logger
from pathlib import Path
from typing import List, Dict, Any

import glob
import os
import logging
import time

from preparator import (DetectionPlace,
                        QueuePutter,
                        QueueGetter)
from cropper_generator import (MPCropperDatasetGenerator,
                               Cropper,
                               print_labels_stats)
from fire_detector import (FireDetector,
                           MPFireDetectorFilter)
from vanilla_detector import (MPVanillaDetector,
                              VanillaDetector,
                              MultiCameraFileGetter,
                              CameraConfigController)

HOME_PATH = str(Path.home())
PATH_TO_IMAGES = os.environ["PATH_TO_IMAGES"]
CONFIGS_PATH = os.environ["CONFIGS_PATH"]
# PATH_TO_IMAGES = "/home/odyssey/mmk_smoke_detection/flow_data/172.30.71.109"
# CONFIG_PATH = "conf/conf_ct131405.json"

PATH_TO_VANILLA_MODEL = "vanilla_model/hatches.h5"
PATH_TO_YOLO = "fire_model/yolov5"
PATH_TO_BEST_WEIGHTS = "fire_model/best.pt"

FIRE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.25

DATASET_RESULT_DIR = os.environ["DATASET_RESULT_DIR"]
QUEUE_SIZE = int(os.environ["QUEUE_SIZE"])

DEBUG_IMAGE = bool(os.environ.get("DEBUG_IMAGE", default=False))

logging.basicConfig(
    # filename=f'dataset_{time.time()}.log',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(f'dataset_{time.time()}.log'),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger(
    name="dataset_logger"
)


def get_images_names(path_to_images: str) -> List[str]:
    return glob.glob(os.path.join(path_to_images, "**", "*.jpg"))


if __name__ == "__main__":
    q1 = Queue(maxsize=QUEUE_SIZE)
    q2 = Queue(maxsize=QUEUE_SIZE)


    images = get_images_names(PATH_TO_IMAGES)
    # images = images[:10]
    LOG.info(f"Total count of images to handle:{len(images)}")

    config_controller = CameraConfigController(
        configs=[
            os.path.join(CONFIGS_PATH, cfg_name)
            for cfg_name in os.listdir(CONFIGS_PATH)
        ]
    )
    vanilla_detector = VanillaDetector(
        emission_threshold=0.003,
        logger=LOG
    )
    
    mp_vanilla_detector = MPVanillaDetector(
        getter=MultiCameraFileGetter(
            pathes_to_images=images,
            logger=LOG
        ),
        putter=QueuePutter(q=q1),
        vanilla_detector=vanilla_detector,
        path_to_model=PATH_TO_VANILLA_MODEL,
        config_controller=config_controller,
        logger=LOG
    )

    mp_fire_detector = MPFireDetectorFilter(
        getter=QueueGetter(
            q=q1
        ),
        putter=QueuePutter(
            q=q2
        ),
        path_to_yolo=PATH_TO_YOLO,
        path_to_weights=PATH_TO_BEST_WEIGHTS,
        fire_detector=FireDetector(),
        fire_threshold=FIRE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        background_prob_threshold=(1 / 18) * 0.25,
        logger=LOG,
        need_debug_image=DEBUG_IMAGE
    )
    cropper = Cropper(
        dataset_main_dir=DATASET_RESULT_DIR,
    )
    mp_cropper = MPCropperDatasetGenerator(
        getter=QueueGetter(
            q=q2
        ),
        cropper=cropper,
        logger=LOG
    )

    start_time = time.time()

    mp_vanilla_detector.start()
    print(f"Start vanilla detector with pid {mp_vanilla_detector.pid}!")
    mp_fire_detector.start()
    print(f"Start fire detector with pid {mp_fire_detector.pid}!")
    mp_cropper.start()
    print(f"Start cropper saver with pid {mp_cropper.pid}!")

    mp_vanilla_detector.join()
    mp_fire_detector.join()
    mp_cropper.join()

    mp_vanilla_detector.terminate()
    mp_fire_detector.terminate()
    mp_cropper.terminate()

    LOG.info(f"Total time of working {time.time() - start_time}")

    print_labels_stats(path_to_dataset=DATASET_RESULT_DIR,
                       extension="jpg")

