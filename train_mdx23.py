from utils.train import train_model
from utils.config_utils import load_config

if __name__ == "__main__":
    device = "cuda:0"
    cfg = load_config("train_config.yaml")
    cfg.model_type = "mdx23c"
    cfg.config_path = "Configs/mdx23c_config.yaml"
    cfg.results_path = "Results/mdx23c"
    cfg.data_path = "Datasets/musdb18hq/train"
    cfg.num_workers = 4
    cfg.valid_path = "Datasets/musdb18hq/valid"
    cfg.seed = 44
    cfg.start_check_point = "Results/model_mdx23c(after prune).ckpt"

    train_model(cfg)
