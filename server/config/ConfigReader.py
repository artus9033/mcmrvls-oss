from logging import getLogger
import os
from typing import Dict

from cerberus import Validator
from utils.constants import serverRootPath
import yaml

from .validationSchemas import configSchema

class ConfigReader:
    def __init__(self) -> None:
        self.__validator = Validator(configSchema)  # type: ignore

        self.__logger = getLogger("ConfigReader")
        self.__logger.setLevel("DEBUG")

    def readConfig(self) -> Dict:
        config = yaml.safe_load(open(os.path.join(serverRootPath, "config.yaml"), "r"))

        self.__logger.debug("Read config file:", config)

        if not self.__validator.validate(config):  # type: ignore
            self.__logger.error("Invalid config file - errors on fields:")

            for field, errors in self.__validator.errors.items():  # type: ignore
                self.__logger.error(f"'{field}': {errors}")

            raise ValueError("Invalid config file")

        self.__logger.debug("Config file is valid")

        return config
