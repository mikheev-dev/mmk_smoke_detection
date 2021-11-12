from config import Config
from typing import Dict

import glob
import os


PATH_TO_CONF = "new_conf"


class ConfigController:
    configs: Dict[str, Config]

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
            conf_data = Config(conf_file)
            self.configs[conf_key] = conf_data

    def __getitem__(
            self,
            key: str
    ) -> Config:
        return self.configs[key]

