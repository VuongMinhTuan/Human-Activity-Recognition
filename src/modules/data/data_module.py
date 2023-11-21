import requests
import zipfile
from pathlib import Path
from transforms import *
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

        self.data_path = Path(data_path)
        self.img_size = img_size
        
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }



    # Check the directory
    def __check_dir(self):
        return True if (self.data_path / "CoffeeStore").is_dir() else False


    # Load data
    def create(self):
        path = self.data_path / "CoffeeStore"
        
        if self.__check_dir():
            print(f"{path} directory exists.")
            return
        
        print(f"Did not find {path} directory, creating one...")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Download coffee store data
        with open(self.data_path / "CoffeeStore.zip", "wb") as f:
            request = requests.get("https://drive.google.com/file/d/1rrZiQroQqxrNgeptIVmGKxCrv27pLl-9/view?usp=drive_link")
            print("Downloading coffee store data...")
            f.write(request.content)
            print("Downloaded the coffee store data successfully!!!")
        
        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(self.data_path / "CoffeeStore.zip", "r") as zip_ref:
            print("Unzipping coffee store data...") 
            zip_ref.extractall(self.data_path / "CoffeeStore")
            print("Unzipped the coffee store data successfully!!!")



data_path = Path("C:/Tuan/GitHub/Human-Activity-Recognition/data")

test = DataModule(data_path)
test.create()