import zipfile, os, requests
from splitfolders import fixed
from pathlib import Path
from typing import Tuple




class DataPreprocessing:
    def __init__(
        self,
        dataset_url: str,
        dataset_path: str,
        ratios: Tuple[float, float , float] = (0.7, 0.15, 0.15)
    ):
        
        self.data_config = {
            "dataset_url": dataset_url,
            "dataset_path" : Path(dataset_path).with_suffix(''),
            "file_name" : dataset_path.split('/')[-1],
            "ratios": ratios
        }


    def __call__(self):
        self.download()
        self.unzip()
        self.split_dataset()


    # Check the directory
    def __check_dir(self):
        return True if os.path.exists(self.data_config['dataset_path']) else False
    

    # Download the dataset from url
    def download(self):
        if self.__check_dir():
            print(f"{self.data_config['dataset_path']} directory has already existed!!!")
            return
        
        with open(self.data_config['dataset_path'].parent, "wb") as f:
            print(f"Downloading {self.data_config['file_name']} dataset...")
            r = requests.get(self.data_config['dataset_url'])
            f.write(r.content)
            print(f"Downloaded {self.data_config['file_name']} dataset successfully!!!")


    # Unzip the dataset file
    def unzip(self):
        if self.__check_dir():
            print(f"{self.data_config['dataset_path']} directory has already existed!!!")
            return


        with zipfile.ZipFile(self.data_config['dataset_path'].with_suffix('.zip'), "r") as zip_ref:
            print(f"Unzipping the {self.data_config['file_name']} dataset...") 
            zip_ref.extractall(self.data_config['dataset_path'].parent)
            print(f"Unzipped the {self.data_config['file_name']} dataset successfully!!!")



    # Create train, test, validation folders
    def split_dataset(self):
        print("Spliting dataset...")

        fixed(
            input= self.data_config['dataset_path'],
            output= self.data_config['dataset_path'].parent.joinpath("dataset"),
            seed= 1337,
            fixed= (350, 300),
            oversample= False,
            group_prefix= None,
            move= False
        )

        print("Splited dataset successfully!!!")