import yaml, os
from rich import print
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from lightning.pytorch import LightningDataModule
from .preprocessing import DataPreprocessing
from .transforms import *




class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        argument_level: int = 0,
        image_size: Tuple[int, int] | int = (224, 224),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        # Initialize the parent class
        super().__init__()
        self.dataset_dir = dataset_dir
        self.argument_level = argument_level
        self.image_size = image_size

        self.loader = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }


        # Load configuration for preprocessing dataset
        with open("C:/Tuan/GitHub/Human-Activity-Recognition/config/data/preprocessing.yaml", 'r') as file:
            preprocess_cfg = yaml.safe_load(file)

        
        # Preprocessing dataset
        preprocess = DataPreprocessing(**preprocess_cfg)
        preprocess("dataset")

        # Classes of datatset
        self.classes = preprocess.classes
    

    def setup(self, stage: str):
        # Check if dataset is created
        if not hasattr(self, "dataset"):
            # List of transformation level
            transform_level = {
                i: getattr(DataTransformation, f"argument_{i}") for i in range(6)
            }

            # Transform configuration
            if self.argument_level not in transform_level:
                raise ValueError(
                    "Use 0 for the default transformation, or scale up to 5 for the strongest effect."
                )
            

            # Training dataset
            self.train_dataset = ImageFolder(
                root= os.path.join(self.dataset_dir, "train"),
                transform= transform_level[self.argument_level](self.image_size)
            )

            # Validation dataset
            self.val_dataset = ImageFolder(
                root= os.path.join(self.dataset_dir, "val"),
                transform= transform_level[self.argument_level](self.image_size)
            )


            # Test dataset
            self.test_dataset = ImageFolder(
                root= os.path.join(self.dataset_dir, "test"),
                transform= transform_level[self.argument_level](self.image_size)
            )


            # Dataset
            self.dataset = ConcatDataset(
                [self.train_dataset, self.val_dataset, self.test_dataset]
            )

        if stage == "fit":
            print(f"[bold]Data path:[/] [green]{self.dataset_dir}[/]")
            print(f"[bold]Number of data:[/] {len(self.dataset):,}")
            print(f"[bold]Number of classes:[/] {len(self.classes):,}")

    
    # Train dataset loader
    def train_dataloader(self):
        return DataLoader(
            dataset= self.train_dataset,
            **self.loader,
            shuffle= True,
        )
    
    # Validation dataset loader
    def val_dataloader(self):
        return DataLoader(
            dataset= self.val_dataset,
            **self.loader,
            shuffle= True
        )
    
    # Test dataset loader
    def test_dataloader(self):
        return DataLoader(
            dataset= self.test_dataset,
            **self.loader,
            shuffle= True
        )