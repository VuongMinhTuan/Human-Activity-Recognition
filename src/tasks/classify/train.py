# Setup root directory
from rootutils import autosetup
autosetup()

import torch
from torch import optim, nn
from torch.optim import lr_scheduler as ls
from modules import DataModule, Module, DataTransformation
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer



def main():
    pass