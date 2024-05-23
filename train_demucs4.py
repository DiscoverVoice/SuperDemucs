import torch
import copy
import numpy as np
from tqdm import tqdm
import os
from utils.prune_utils.concern_identification import ConcernIdentification
from utils.prune_utils.weight_remover import WeightRemover
from utils.dataset import download_musdb, load_data
from utils.train import (
    train_model,
)
from utils.config_utils import load_config

if __name__ == "__main__":
    device = "cuda:0"
    cfg = load_config("train_config.yaml")
    cfg.model_type = "htdemucs"
    cfg.config_path = "Configs/htdemucs_config.yaml"
    cfg.results_path = "Results/"
    cfg.data_path = "Datasets/musdb18hq/train"
    cfg.num_workers = 4
    cfg.valid_path = "Datasets/musdb18hq/valid"
    cfg.seed = 44
    cfg.start_check_point = "Results/model_htdemucs(after prune).ckpt"

    train_model(cfg)
