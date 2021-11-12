import sys
sys.path.insert(1, "/home/odyssey/mmk_smoke_detection")

from dataset_preparator.preparator import plot_detections
from dataset_preparator.classJson import JsonData
from dataset_preparator.cropper_generator import Cropper
from typing import Dict


import glob
import os

# ct131401  ct131402  ct131403  ct131404	ct131405  ct131406  ct1314.csv

PATH_TO_CONF = "/home/odyssey/mmk_smoke_detection/validation/new_conf"
PATH_TO_VALIDATION_DATA = "/home/odyssey/mmk_smoke_detection/validation/05.10.21"

validation_camera_keys = [
    path.split('/')[1]
    for path in glob.glob(os.path.join(PATH_TO_VALIDATION_DATA, '*'))
    if os.path.isdir(path)
]


class ConfigController:
    configs: Dict[str, JsonData]
        
    @staticmethod
    def extract_key_from_conf_filename(
        conf_filename: str
    ) -> str:
        return os.path.basename(conf_filename)[5:13]
    
    def __init__(
        self,
        path_to_configs: str,
    ):
        self.configs = {}
        conf_files = glob.glob(os.path.join(path_to_configs, '*'))
        print(conf_files)
        for conf_file in conf_files:
            conf_key = self.extract_key_from_conf_filename(conf_file)
            conf_data = JsonData(conf_file)
            self.configs[conf_key] = conf_data
            
    def __getitem__(
        self,
        key: str
    ) -> JsonData:
        return self.configs[key]

