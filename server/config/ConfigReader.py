from logging import getLogger
import os
from typing import Dict

from cerberus import Validator
from utils.constants import serverRootPath
import yaml

from .merge import deep_merge_config
from .validationSchemas import configSchema


class ConfigReader:
    def __init__(self) -> None:
        self.__validator = Validator(configSchema)  # type: ignore

        self.__logger = getLogger("ConfigReader")
        self.__logger.setLevel("DEBUG")

    @staticmethod
    def get_scenario_config_path(playback_scenario: str) -> str:
        return os.path.join(serverRootPath, "res", "video", playback_scenario, "config.yaml")

    def readConfig(self, playback_scenario: str | None = None) -> Dict:
        config_path = os.path.join(serverRootPath, "config.yaml")
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        if playback_scenario is not None:
            scenario_config_path = self.get_scenario_config_path(playback_scenario)
            if os.path.exists(scenario_config_path):
                with open(scenario_config_path, "r") as scenario_config_file:
                    scenario_config = yaml.safe_load(scenario_config_file) or {}

                if not isinstance(scenario_config, dict):
                    raise ValueError(f"Invalid scenario config file (expected mapping): {scenario_config_path}")

                config = deep_merge_config(config, scenario_config)
                self.__logger.info(
                    "Applied playback scenario config overrides from %s",
                    scenario_config_path,
                )

        self.__logger.debug("Read config file:", config)

        if not self.__validator.validate(config):  # type: ignore
            self.__logger.error("Invalid config file - errors on fields:")

            for field, errors in self.__validator.errors.items():  # type: ignore
                self.__logger.error(f"'{field}': {errors}")

            raise ValueError("Invalid config file")

        self.__logger.debug("Config file is valid")

        return config
