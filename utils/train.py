from torch.utils.tensorboard import SummaryWriter
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from ml_collections import ConfigDict
import yaml

import random
import time
import copy
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


from utils.paths import p
from utils.dataset import MSSDataset
from utils.audio_utils import demix_track, demix_track_demucs, sdr
from utils.models.bs_roformer import MelBandRoformer
from utils.models.bs_roformer import BSRoformer

def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def masked_loss(y_, y, q, coarse=True):
    # shape = [num_sources, batch_size, num_channels, chunk_size]
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()

def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    if model_type == 'mdx23c':
        from utils.models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from utils.models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from utils.models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from utils.models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        model = BSRoformer(
            **dict(config.model)
        )
    elif model_type == 'swin_upernet':
        from utils.models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from utils.models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(
            **config.model
        )
    elif model_type == 'scnet_unofficial':
        from utils.models.scnet_unofficial import SCNet
        model = SCNet(
            **config.model
        )
    elif model_type == 'scnet':
        from utils.models.scnet import SCNet
        model = SCNet(
            **config.model
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model, config

def load_not_compatible_weights(model, weights, verbose=False):
    new_model = model.state_dict()
    old_model = torch.load(weights)
    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']

    for el in new_model:
        if el in old_model:
            if verbose:
                print('Match found for {}!'.format(el))
            if new_model[el].shape == old_model[el].shape:
                if verbose:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if verbose:
                        print('Shape is different: {} != {}'.format(tuple(new_model[el].shape), tuple(old_model[el].shape)))
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if verbose:
                print('Match not found for {}!'.format(el))
    model.load_state_dict(
        new_model
    )

def valid(model, config, model_config, device, verbose=False):
    # For multiGPU extract single model

    model.eval()
    all_mixtures_path = []
    base_valid_path = config.valid_path
    subdirs = next(os.walk(base_valid_path))[1]

    for subdir in subdirs:
        part = sorted(glob.glob(os.path.join(base_valid_path, subdir, '*mixture.wav')))
        if len(part) == 0:
            print(f'No validation data found in: {os.path.join(base_valid_path, subdir)}')
        all_mixtures_path.extend(part)

    if verbose:
        print('Total mixtures: {}'.format(len(all_mixtures_path)))

    instruments = model_config.training.instruments
    if model_config.training.target_instrument is not None:
        instruments = [model_config.training.target_instrument]

    all_sdr = dict()
    for instr in model_config.training.instruments:
        all_sdr[instr] = []

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    pbar_dict = {}
    for path in all_mixtures_path:
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        if verbose:
            print('Song: {}'.format(os.path.basename(folder)))
        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if config.model_type == 'htdemucs':
            res = demix_track_demucs(model_config, model, mixture, device)
        else:
            res = demix_track(model_config, model, mixture, device)
        for instr in instruments:
            if instr != 'other' or model_config.training.other_fix is False:
                track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
            else:
                # other is actually instrumental
                track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                track = mix - track
            # sf.write("{}.wav".format(instr), res[instr].T, sr, subtype='FLOAT')
            references = np.expand_dims(track, axis=0)
            estimates = np.expand_dims(res[instr].T, axis=0)
            sdr_val = sdr(references, estimates)[0]
            if verbose:
                print(instr, res[instr].shape, sdr_val)
            all_sdr[instr].append(sdr_val)
            pbar_dict['sdr_{}'.format(instr)] = sdr_val
        if not verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    sdr_avg = 0.0
    for instr in instruments:
        sdr_val = np.array(all_sdr[instr]).mean()
        print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
        sdr_avg += sdr_val
    sdr_avg /= len(instruments)
    if len(instruments) > 1:
        print('SDR Avg: {:.4f}'.format(sdr_avg))
    return sdr_avg

def train_model(cfg: DictConfig):
    manual_seed(cfg.seed + int(time.time()))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Fix possible slow down with dilation convolutions
    torch.multiprocessing.set_start_method('spawn')

    model, model_config = get_model_from_config(cfg.model_type, cfg.config_path)
    print("Instruments: {}".format(model_config.training.instruments))

    if not os.path.isdir(cfg.results_path):
        os.mkdir(cfg.results_path)

    use_amp = model_config.training.get('use_amp', True)
    writer = SummaryWriter(cfg.results_path)
    device_ids = cfg.device_ids
    batch_size = model_config.training.batch_size * len(device_ids)

    trainset = MSSDataset(
        model_config,
        cfg.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(cfg.results_path, f'metadata_train.pkl'),
        dataset_type=cfg.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    if cfg.start_check_point:
        print('Start from checkpoint: {}'.format(cfg.start_check_point))
        if 1:
            load_not_compatible_weights(model, cfg.start_check_point, verbose=False)
        else:
            model.load_state_dict(
                torch.load(cfg.start_check_point)
            )

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            print('Use single GPU: {}'.format(device_ids))
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            print('Use multi GPU: {}'.format(device_ids))
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run training on CPU. It will be very slow...')
        model = model.to(device)

    optimizer = None
    if model_config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=model_config.training.lr)
    elif model_config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=model_config.training.lr)
    elif model_config.training.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=model_config.training.lr)
    elif model_config.training.optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=model_config.training.lr, momentum=0.999)
    else:
        print('Unknown optimizer: {}'.format(model_config.training.optimizer))
        exit()

    gradient_accumulation_steps = model_config.training.get('gradient_accumulation_steps', 1)

    print("Patience: {} Reduce factor: {} Batch size: {} Grad accum steps: {} Effective batch size: {}".format(
        model_config.training.patience,
        model_config.training.reduce_factor,
        batch_size,
        gradient_accumulation_steps,
        batch_size * gradient_accumulation_steps,
    ))

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=model_config.training.patience, factor=model_config.training.reduce_factor)

    if cfg.use_multistft_loss:
        loss_options = model_config.get('loss_multistft', {})
        print('Loss options: {}'.format(loss_options))
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(
            **loss_options
        )

    scaler = torch.cuda.amp.GradScaler()
    print('Train for: {}'.format(model_config.training.num_epochs))
    best_sdr = -100
    for epoch in range(model_config.training.num_epochs):
        model.train().to(device)
        print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(device)
            x = mixes.to(device)  # mixture

            with torch.cuda.amp.autocast(enabled=use_amp):
                if cfg.model_type in ['mel_band_roformer', 'bs_roformer']:
                    loss = model(x, y)
                    if type(device_ids) != int:
                        loss = loss.mean()
                else:
                    y_ = model(x)
                    if cfg.use_multistft_loss:
                        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
                        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
                        loss = loss_multistft(y1_, y1)
                        if cfg.use_mse_loss:
                            loss += 1000 * nn.MSELoss()(y1_, y1)
                        if cfg.use_l1_loss:
                            loss += 1000 * F.l1_loss(y1_, y1)
                    elif cfg.use_mse_loss:
                        loss = nn.MSELoss()(y_, y)
                    elif cfg.use_l1_loss:
                        loss = F.l1_loss(y_, y)
                    else:
                        loss = masked_loss(
                            y_,
                            y,
                            q=model_config.training.q,
                            coarse=model_config.training.coarse_loss_clip
                        )

            loss /= gradient_accumulation_steps
            scaler.scale(loss).backward()
            if model_config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), model_config.training.grad_clip)

            if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            li = loss.item() * gradient_accumulation_steps
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            loss.detach()

            writer.add_scalar('Loss/train', li, epoch * len(train_loader) + i)

        print('Training loss: {:.6f}'.format(loss_val / total))

        store_path = os.path.join(cfg.results_path, f'last_{cfg.model_type}.ckpt')
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        torch.save(state_dict, store_path)

        sdr_avg = valid(model, cfg, model_config, device, verbose=False)

        if sdr_avg > best_sdr:
            store_path = os.path.join(cfg.results_path, f'model_{cfg.model_type}_ep_{epoch}_sdr_{sdr_avg:.4f}.ckpt')
            print('Store weights: {}'.format(store_path))
            state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
            torch.save(state_dict, store_path)
            best_sdr = sdr_avg
        scheduler.step(sdr_avg)

        writer.add_scalar('SDR/valid', sdr_avg, epoch)

    writer.close()