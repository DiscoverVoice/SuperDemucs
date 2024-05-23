import torch
import copy
import numpy as np
import os
import logging
from utils.prune_utils.concern_identification import ConcernIdentification
from utils.prune_utils.weight_remover import WeightRemover
from utils.dataset import load_data
from utils.train import valid, load_not_compatible_weights, get_model_from_config
from utils.config_utils import load_config
from utils.paths import p
from utils.logger import stdio2logs

logging.basicConfig(filename=os.path.join(p.Logs, 'pruning_demucs.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.multiprocessing.set_start_method('spawn')

if __name__ == "__main__":
    stdio2logs()
    logging.info("Starting the pruning script")
    device = 'cuda:0'
    cfg = load_config('train_config.yaml')
    cfg.model_type = 'htdemucs'
    cfg.config_path = 'Configs/htdemucs_config.yaml'
    cfg.results_path = 'Results/htdemucs'
    cfg.data_path = 'Datasets/musdb18hq/train'
    cfg.num_workers = 0
    cfg.valid_path = 'Datasets/musdb18hq/valid'
    cfg.seed = 44
    cfg.start_check_point = 'Results/model_htdemucs(before prune).ckpt'

    logging.info("Loading model and configuration")
    htdemucs, htdemucs_config = get_model_from_config(cfg.model_type, cfg.config_path)
    use_amp = htdemucs_config.training.get('use_amp', True)
    load_not_compatible_weights(htdemucs, cfg.start_check_point, verbose=False)
    htdemucs = htdemucs.to(device)

    logging.info("Loading data")
    train_dataloader, valid_dataloader, test_dataloader = load_data(htdemucs_config, cfg, 2)

    ref_htdemucs = copy.deepcopy(htdemucs)
    weight_remover = WeightRemover(htdemucs, device, 0.9)

    logging.info("Start pruning")
    for idx in range(10):
        tr = torch.tensor(np.random.rand(2, 2, 485100), dtype=torch.float32).to(device)
        with torch.no_grad():
            y_ = weight_remover.process(tr)
        weight_remover.apply_removal()
    valid(htdemucs, cfg, htdemucs_config, device)

    logging.info("Concern Identification")
    ci = ConcernIdentification(ref_htdemucs, htdemucs, device, 0.7)
    _, temp_config = get_model_from_config(cfg.model_type, cfg.config_path)
    temp_config.training.instruments = ["vocals"]
    temp_dataloader, _, _ = load_data(temp_config, cfg, 2)

    for i, (batch, mixes) in enumerate(temp_dataloader):
        y = batch.to(device)
        x = mixes.to(device)
        with torch.no_grad():
            ci.process(x)
        ci.apply_prune()
        if i > 100:
            break

    store_path = os.path.join(cfg.results_path, f'model_htdemucs(after prune).ckpt')
    state_dict = htdemucs.state_dict()
    torch.save(state_dict, store_path)
    logging.info("Pruning script finished successfully")
