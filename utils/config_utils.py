import yaml
import os
import sys
from hydra import initialize, compose
from omegaconf import DictConfig

from utils.paths import p


def load_config(config_name):
    # Absolute path to the configuration directory
    config_dir_abs = '/home/Minwoo/Github/DiscoverVoice/SuperDemucs/Configs'

    # Compute the relative path from the current working directory
    config_dir_rel = '../Configs'

    config_file = os.path.join(config_dir_abs, config_name)

    # Use the relative path in initialize
    with initialize(config_path=config_dir_rel):
        config = compose(config_name=os.path.basename(config_file))
    return config