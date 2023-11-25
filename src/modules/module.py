import torch, os
from .utils import device_handler
from typing import Dict
from torch import nn, optim
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule




class Module(LightningModule):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module = None,
            optimizer: optim.Optimizer | Dict = None,
            scheduler: optim.Optimizer = None,
            checkpoint: str = None,
            device: str = "auto"
        ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Check checkpoint
        if checkpoint:
            self.load(checkpoint, device= device)

    
    def forward(self, X):
        return self.model(X)
    

    def configure_optimizers(self):
        if not self.scheduler:
            return self.optimizer
        
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]

        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]

        return self.optimizer, self.scheduler
    

    def _log(self, stage: str, loss, y_hat, y):
        acc = accuracy(
            preds= y_hat, target= y, task= "multiclass", num_classes= self.model.num_classes
        )

        self.log_dict(
            dictionary= {f"{stage}/loss": loss, f"{stage}/accuracy": acc},
            on_step= False,
            on_epoch= True,
        )

    
    # Create train, validation, test steps
    def training_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage= "train", loss= loss, y_hat= y_hat, y= y)
        
        return loss


    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage= "val", loss= loss, y_hat= y_hat, y= y)


    def test_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage= "test", loss= loss, y_hat= y_hat, y= y)


    # Load the checkpoint
    def load(
        self,
        path: str,
        strict: bool= True, 
        device: str= "auto", 
        verbose: bool= True
    ):
        
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        print("[bold]Load checkpoint:[/] Loading...") if verbose else None

        self.load_state_dict(
            state_dict= torch.load(
                path,
                map_location= device_handler(device)
            )["state_dict"],

            strict= strict,
        )

        print("[bold]Load checkpoint:[/] Done") if verbose else None


    # Save parameter
    def save_hparams(self, config: Dict):
        self.hparams.update(config)
        self.save_hyperparameters()