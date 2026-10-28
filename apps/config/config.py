import os
import yaml
from apps import singleton


@singleton
class AppConfig(dict):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        for key, value in kwargs.items():
            self[key] = value


APP_CONFIG = None


def init_config(profile: str):
    global APP_CONFIG
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, f"application-{profile}.yml")
    if not os.path.exists(yaml_path):
        # 兼容未指定环境时默认 dev
        yaml_path = os.path.join(cur_path, "application-dev.yml")
    with open(yaml_path, encoding="UTF-8") as fs:
        configs = yaml.safe_load(fs)
    APP_CONFIG = AppConfig(configs)


def get_app_config() -> AppConfig:
    global APP_CONFIG
    if APP_CONFIG is None:
        init_config("dev")
    return APP_CONFIG
