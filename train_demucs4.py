from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.model_utils import LightningWrapper, load_model
from utils.config_utils import load_config
import pytorch_lightning as pl
import torch


if __name__ == "__main__":
    config = load_config("demucs4_config.yaml")
    model = load_model(config)

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

    pl_model = LightningWrapper(model, config)
    trainer.fit(pl_model, train_loader, val_loader)
    torch.save(pl_model.state_dict(), 'final.pt')
    pl_model.load_state_dict(torch.load('final.pt'))