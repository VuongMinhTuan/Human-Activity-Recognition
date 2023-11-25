import os
from transforms import *
from typing import Tuple, List
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from preprocessing import DataPreprocessing
from src.modules.utils import workers_handler




class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        dataset_url = None,
        image_size: Tuple[int, int] | list = (224, 224),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        argument_level: int = 0
    ):
        
        # Initialize the parent class
        super().__init__()

        self.argument_level = argument_level

        self.data_config = {
            "dataset_path" : dataset_path,
            "dataset_url": dataset_url,
            "image_size" : image_size
        }
        
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
        }


    
    # Get classes
    @property
    def classes(self) -> List[str]:
        return sorted(os.listdir(os.path.join(self.data_config['dataset_path'], "train")))
    

    # Check data directory is existed or not
    def __check_dir(self):
        return True if not os.path.exists(self.data_config['data_path']) else False
    

    def setup(self, stage: str):
        if not self.__check_dir():
            prepare = DataPreprocessing(
                dataset_url= "https://drive.google.com/file/d/1USFmkMyZ0bRCKcuzq9kjdQiCkfbEC1iZ/view?usp=sharing" 
                if self.data_config['dataset_url'] is None 
                else self.data_config['dataset_url'],
                
                dataset_path= self.data_config['dataset_path'],
                ratios= (0.7, 0.15, 0.15)
            )

            prepare()


        # Get all transform levels
        transfrom_levels = {
            i: getattr(DataTransformation, f"argument_lv{i}") for i in range(6)
        }


        if self.argument_level not in transfrom_levels:
            raise ValueError (
                "Use 0 for the default transformation or scale up to 5 for the strongest effect"
            )
        

        # Create dataset included train, validation and test dataset
        self.train_data = ImageFolder(
            root= os.path.join(self.data_config['dataset_path'], "train"),
            transform= transfrom_levels[self.argument_level](self.data_config['image_size'])
        )

        self.val_data = ImageFolder(
            root= os.path.join(self.data_config['dataset_path'], "val"),
            transform= transfrom_levels[self.argument_level](self.data_config['image_size'])
        )

        self.test_data = ImageFolder(
            root= os.path.join(self.data_config['dataset_path'], "test"),
            transform= transfrom_levels[self.argument_level](self.data_config['image_size'])
        )

        self.dataset = ConcatDataset(
            [self.train_data, self.val_data, self.test_data]
        )

        if stage == "fit":
            print(f"[bold]Data path:[/] [green]{self.data_config['data_path']}[/]")
            print(f"[bold]Number of data:[/] {len(self.dataset):,}")
            print(f"[bold]Number of classes:[/] {len(self.classes):,}")


    def train_dataloader(self):
        return DataLoader(dataset= self.train_data, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset= self.val_data, **self.loader_config, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset= self.test_data, **self.loader_config, shuffle=False)