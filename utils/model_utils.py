import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl


import torchmetrics
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from ..models.bs_roformer import BSRoformer
from ..models.mdx23.tfc_tdf import TFC_TDF_net
from ..models.demucs4.htdemucs import HTDemucs


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, config, criterion):
        super(LightningWrapper, self).__init__()
        self.model = None
        self.config = None
        self.criterion = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        audio_data, sample_rate, num_samples, track_name = batch
        outputs = self.model(audio_data)
        loss = self.criterion(outputs, audio_data)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_data, sample_rate, num_samples, track_name = batch
        outputs = self.model(audio_data)
        loss = self.criterion(outputs,audio_data)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['model']['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config['model']['T_O'],
                                                T_mult=self.config['model']['T_mult'],
                                                eta_min=self.config['model']['eta_min'])
        return [optimizer], [scheduler]


def load_model(config):
    model_name = config["model"]["name"]
    if model_name == "demucs4":
        from ..models.demucs4.htdemucs import get_model
        model = get_model(config)
    elif model_name == "mdx23":
        model = TFC_TDF_net(config)
    elif model_name == "bs_roformer":
        model = BSRoformer(**dict(config.model))
    else:
        raise ValueError("Model is not selected")
    weight_path = config["model"]["weight_path"]
    model.load_state_dict(torch.load(weight_path))
    return model