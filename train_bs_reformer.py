from utils.config_utils import load_config
from utils.paths import Paths
from utils.dataset_util import load_musdb

if __name__ == "__main__":
    data_config = load_config("bs_reformer_config")
    train_dataloader, val_dataloader, test_dataloader = load_musdb(config = data_config)
