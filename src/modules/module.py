import torch, os
from typing import Dict
from torch import nn, optim
from rich import print
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule
from .utils import device_handler



class LitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer | Dict = None,
        scheduler: optim.Optimizer = None,
        checkpoint: str = None,
        device: str = "auto"
    ):
        # Initialize parent class
        super().__init__()


        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if checkpoint:
            self.load(checkpoint, device= device)


    # Run dataset through model only
    def forward(self, X):
        return self.model(X)


    # Define optimizers and schedulers
    def configure_optimizers(self):
        if not self.scheduler:
            return self.optimizer
        
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]

        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]

        return (self.optimizer, self.scheduler)
    

    def log(self, stage: str, loss, y_hat, y):
        acc = accuracy(
            preds= y_hat,
            target= y,
            task= "multiclass",
            num_classes= self.model.num_classes
        )

        self.log_dict(
            dictionary= {f"{stage}/loss": loss, f"{stage}/accuracy": acc},
            on_step= True,
            on_epoch= True
        )
    

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        
        return loss
    

    def validation_step(self, bacth, batch_idx):
        X, y = bacth
        y_hat = self(X)
        loss = self.criterion(y_hat, y)

        self.log("validation", loss, y_hat, y)


    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)

        self.log("test", loss, y_hat, y)


    # Load checkpoint
    def load(
        self,
        path: str,
        strict: bool = True,
        device: str = "auto",
        verbose: bool = True
    ):
        
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        print("[bold][green]Loading checkpoint...[/]") if verbose else None

        self.load_state_dict(
            state_dict= torch.load(
                path,
                map_location= device_handler(device)
            )["state_dict"],
            
            strict=strict,
        )

        print("[bold][green]Load checkpoint successfully!!!") if verbose else None

    
    # Save hyperparameters
    def save_hparams(self, config: Dict) -> None:
        self.hparams.update(config)
        self.save_hyperparameters()