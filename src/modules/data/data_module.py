from .transforms import *
from typing import Tuple
from pytorch_lightning import LightningDataModule




class DataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        img_size: Tuple[int, int] | list = (224, 224),
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        
        # Initialize the parent class
        super().__init__()

        self.data_path = data_path
        self.img_size = img_size
        
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }



    # Load data
    def load(self):
        pass