# Setup root directory
from rootutils import autosetup
autosetup()

from rich import traceback
traceback.install()

import torch
from torch import nn, optim
from typing import Dict
from src.models import VIT
from src.modules import DataModule, LitModule, scheduler_with_warmup, load_yaml, custom_callbacks
from lightning.pytorch import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger



def main(cfg: Dict):
    # Set precision
    torch.set_float32_matmul_precision("high")


    # Set seed
    if cfg["set_seed"]:
        seed_everything(seed= cfg["set_seed"], workers= True)


    # Define dataset
    DATASET = DataModule(
        **cfg["data"]
    )


    # Define model
    MODEL = VIT(
        num_classes= len(DATASET.classes),
        **cfg["model"]
    )


    # Set up loss and optimizer
    LOSS = nn.CrossEntropyLoss()

    OPTIMIZER = optim.AdamW(
        MODEL.parameters(),
        lr= cfg["trainer"]["learning_rate"],
        weight_decay= cfg["trainer"]["learning_rate"]
    )


    # Set up scheduler
    SCHEDULER = scheduler_with_warmup(
        scheduler= optim.lr_scheduler.CosineAnnealingLR(
            optimizer= OPTIMIZER,
            T_max= cfg["trainer"]["num_epoch"]
        ),

        **cfg["scheduler"]
    )


    # Module
    LIT_MODULE = LitModule(
        model= MODEL,
        criterion= LOSS,
        optimizer= OPTIMIZER,
        scheduler= SCHEDULER,
        checkpoint= cfg["trainer"]["checkpoint"]
    )


    # Save hyperparameters
    LIT_MODULE.save_hparams(cfg)

    # Trainer
    TRAINER = Trainer(
        max_epochs= cfg["trainer"]["num_epoch"],
        precision= cfg["trainer"]["precision"],
        callbacks= custom_callbacks(),
        logger= TensorBoardLogger(save_dir= "C:/Tuan/GitHub/Human-Activity-Recognition/logs")
    )

    # Training
    TRAINER.fit(LIT_MODULE, DATASET)

    # Testing
    TRAINER.test(LIT_MODULE, DATASET)


if __name__ == "__main__":
    main(load_yaml("C:/Tuan/GitHub/Human-Activity-Recognition/config/classifiers/train.yaml"))