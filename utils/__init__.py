# coding: utf-8
# write reference: https://timothybramlett.com/How_to_create_a_Python_Package_with___init__py.html

import warnings
from functools import partial
from pathlib import Path
from typing import Tuple

from .log import get_logger
from .scheduler_task import Periodic, Scheduler, on_exception
from .util import dict_merge, dict_nested_get, load_config

__version__ = "0.0.5"

__all__: Tuple[str] = (
    "config",
    "config_nested_get",
    "dict_merge",
    "dict_nested_get",
    "get_logger",
    "load_config",
    "on_exception",
    "pyproject_logger_setting",
    "Periodic",
    "Scheduler",
)

try:
    config = load_config()  # noqa:F405

    # Always try load .env.toml
    env_file_path = "./.env.toml"
    if Path(env_file_path).is_file():
        print(f"Load '{env_file_path}' success")
        env_config = load_config(env_file_path)
        if env_config:
            config = dict_merge(config, env_config)

    env_file_path_list = dict_nested_get(config, "env.file_path", [])
    env_file_path_list = [env_file_path_list] if isinstance(env_file_path_list, str) else env_file_path_list

    for env_file_path in env_file_path_list:
        if not Path(env_file_path).is_file():
            raise FileNotFoundError(f"File '{env_file_path}' not exists or not is a file.")

        env_config = load_config(env_file_path)
        if env_config:
            config = dict_merge(config, env_config)

    logger_setting = config.get("logger", {})
    log_level = logger_setting.get("log_level", "INFO")
    log_fmt = logger_setting.get("log_fmt", "STD")
    log_path = logger_setting.get("log_path", "./log")
    log_count = logger_setting.get("log_count", 7)
    max_bytes = logger_setting.get("max_bytes", 10)
    warning_level = logger_setting.get("warning_level", None)

    if warning_level:
        warnings.simplefilter(warning_level)  # Change the filter in this process

    pyproject_logger_setting = {
        "log_level": log_level,
        "log_fmt": log_fmt,
        "log_path": log_path,
        "log_count": log_count,
        "max_bytes": max_bytes,
    }
except Exception as e:
    config = dict()
    warnings.warn(e, Warning)

config_nested_get = partial(dict_nested_get, config)
