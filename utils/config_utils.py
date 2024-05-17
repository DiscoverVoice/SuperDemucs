import yaml
import os
import sys
from paths import p


def load_config(config_name):
    config_dir = p.config_dir
    config_file = config_dir / config_name
    with config_file.open() as file:
        config = yaml.safe_load(file)
        return config
