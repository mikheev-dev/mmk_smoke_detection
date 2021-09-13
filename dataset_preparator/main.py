from multiprocessing import Queue
from pathlib import Path
from typing import List, Dict, Any
from tensorflow import keras

import glob
import os
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

FIRE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.25

DATASET_RESULT_DIR = os.environ["DATASET_RESULT_DIR"]
QUEUE_SIZE = int(os.environ["QUEUE_SIZE"])


def get_images_names(path_to_images: str) -> List[str]:
    return glob.glob(os.path.join(path_to_images, "**", "*.jpg"))


if __name__ == "__main__":
    q1 = Queue(maxsize=QUEUE_SIZE)
    q2 = Queue(maxsize=QUEUE_SIZE)

    images = get_images_names(PATH_TO_IMAGES)
    # images = images[:3000]
    print("Total count of images to handle:", len(images))

    config_controller = CameraConfigController(
        configs=[
            os.path.join(CONFIGS_PATH, cfg_name)
            for cfg_name in os.listdir(CONFIGS_PATH)
        ]
    )
    vanilla_detector = VanillaDetector(
        emission_threshold=0.003,
        background_prob_threshold=(1 / 18) * 0.1
    )
    
    mp_vanilla_detector = MPVanillaDetector(
        getter=MultiCameraFileGetter(
            pathes_to_images=images
        ),
        putter=QueuePutter(q=q1),
        vanilla_detector=vanilla_detector,
        path_to_model=PATH_TO_VANILLA_MODEL,
        config_controller=config_controller
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
        iou_threshold=IOU_THRESHOLD
    )
    cropper = Cropper(
        dataset_main_dir=DATASET_RESULT_DIR,
    )
    mp_cropper = MPCropperDatasetGenerator(
        getter=QueueGetter(
            q=q2
        ),
        cropper=cropper
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

    print("Total time of working", time.time() - start_time)

    print_labels_stats(path_to_dataset=DATASET_RESULT_DIR,
                       extension="jpg")

