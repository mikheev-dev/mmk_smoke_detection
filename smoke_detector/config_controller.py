from smoke_detector.config import Config, CameraConfig
from typing import Dict, List, Optional, Iterator

import glob
import os


PATH_TO_CONF = "new_conf"


class ConfigController:
    configs: Dict[str, Config]
    camera_configs: Dict[str, CameraConfig]
    _zone_names: List[str]

    @property
    def zone_names(self) -> List[str]:
        return self._zone_names

    @zone_names.setter
    def zone_names(
            self,
            zone_names: List[str]
    ) -> None:
        if len(self.configs.keys()) != len(zone_names):
            print(len(self.configs.keys()), len(zone_names))
            raise Exception("Incosistent number of config keys and number of zone names")
        self._zone_names = zone_names

    @staticmethod
    def extract_key_from_conf_filename(
            conf_filename: str
    ) -> str:
        return os.path.basename(conf_filename)[5:13]

    @property
    def ip_to_conf(self) -> Dict[str, Config]:
        return {
            cfg.camera_ip: cfg
            for _, cfg in self.configs.items()
        }

    def __init__(
            self,
            path_to_configs: Optional[str] = None,
            config_file_pathes: Optional[List[str]] = None
    ):
        if not path_to_configs and not config_file_pathes:
            raise Exception("No configs!")

        self.configs = {}
        self._zone_names = []
        if not config_file_pathes:
            conf_files = glob.glob(os.path.join(path_to_configs, '*'))
            print(conf_files)
        else:
            conf_files = config_file_pathes
        self.camera_configs = {}
        for conf_file in conf_files:
            conf_key = self.extract_key_from_conf_filename(conf_file)
            conf_data = Config(conf_file)
            self.configs[conf_key] = conf_data
            # self.camera_configs[conf_key] = CameraConfig(conf_data)

    def __getitem__(
            self,
            key: str
    ) -> Config:
        return self.configs[key]

    def __iter__(self) -> Iterator:
        if self.zone_names:
            return zip(self.configs.keys(), self.configs.values(), self.zone_names)
        return zip(self.configs.keys(), self.configs.values())

