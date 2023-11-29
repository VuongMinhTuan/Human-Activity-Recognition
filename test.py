from src.modules.data.preprocessing import DataPreprocessing
from src.modules.utils import load_yaml


config = load_yaml("C:/Tuan/GitHub/Human-Activity-Recognition/config/data/video.yaml")

data = DataPreprocessing(**config)

data()
