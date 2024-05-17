import time
import random
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.integration import PyTorchLightningPruningCallback

class LightningWrapper(pl.LightningModule):
    def __init__(self):
        super(LightningWrapper, self).__init__()
        self.model = None
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def create_model(self, model):
        self.model = model
    def training_step(self, batch, batch_idx):
        audio_data, sample_rate, num_samples, track_name = batch
        outputs = self.model()
        loss = self.criterion(outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_data, sample_rate, num_samples, track_name = batch
        outputs = self.model()
        loss = self.criterion(outputs, )
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=config['model']['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['model']['T_O'],
                                                T_mult=config['model']['T_mult'],
                                                eta_min=config['model']['eta_min'])
        return [optimizer], [scheduler]


logger = TensorBoardLogger("tb_logs", name=config['model']['name'])

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=config['model']['save_checkpoint_path'],
    save_top_k=3,
    mode='min',
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=config['model']['max_epochs'],
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    precision=config['model']['precision'],
    accumulate_grad_batches=config['model']['accumulate_grad_batches']
)

pl_model = LightningWrapper()
pl_model.create_model(model)
trainer.fit(pl_model, train_loader, val_loader)
torch.save(pl_model.state_dict(), 'final.pt')
pl_model.load_state_dict(torch.load('final.pt'))