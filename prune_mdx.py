import torch
import copy
import numpy as np
from tqdm import tqdm
import os
from utils.prune_utils.concern_identification import ConcernIdentification
from utils.prune_utils.weight_remover import WeightRemover
from utils.dataset import download_musdb, load_data
from utils.train import train_model, valid, load_not_compatible_weights, get_model_from_config
from utils.config_utils import load_config
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Fix possible slow down with dilation convolutions
torch.multiprocessing.set_start_method('spawn')
train_data, test_data = download_musdb()

device = 'cuda:0'
cfg = load_config('train_config.yaml')
cfg.model_type = 'mdx23c'
cfg.config_path = 'Configs/mdx23c_config.yaml'
cfg.results_path = 'Results/'
cfg.data_path = 'Datasets/musdb18hq/train'
cfg.num_workers = 0
cfg.valid_path = 'Datasets/musdb18hq/valid'
cfg.seed = 44
cfg.start_check_point = 'Results/model_mdx23c(before prune).ckpt'

mdx23c, mdx23c_config = get_model_from_config(cfg.model_type, cfg.config_path)
load_not_compatible_weights(mdx23c, cfg.start_check_point, verbose=False)
mdx23c = mdx23c.to(device)

train_dataloader, valid_dataloader, test_dataloader = load_data(mdx23c_config, cfg, 2)

mdx23c = mdx23c.to(device)
ref_mdx23c = copy.deepcopy(mdx23c)
weight_remover = WeightRemover(mdx23c, device, 0.9)

print("start pruning")
for idx in range(10):
    tr = torch.tensor(np.random.rand(2, 2, 261120), dtype=torch.float32).to(device)
    with torch.no_grad():
        y_ = weight_remover.process(tr)
    weight_remover.apply_removal()
# valid(mdx23c, cfg2, mdx23c_config, device)

print("CI")
ci = ConcernIdentification(ref_mdx23c, mdx23c, device, 0.4)
_, temp_config = get_model_from_config(cfg.model_type, cfg.config_path)
temp_config.training.instruments = ["vocals"]
temp_dataloader, _, _ = load_data(temp_config, cfg, 2)

pbar = tqdm(temp_dataloader)
for i, (batch, mixes) in enumerate(pbar):
    y = batch.to(device)
    x = mixes.to(device)
    with torch.no_grad():
        ci.process(x)
    ci.apply_prune()
    if i > 100:
        break

store_path = os.path.join(cfg.results_path, f'model_mdx23c(after prune).ckpt')
state_dict = mdx23c.state_dict()
torch.save(state_dict, store_path)
