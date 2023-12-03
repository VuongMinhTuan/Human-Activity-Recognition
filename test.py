from src.modules.data.preprocessing import DataPreprocessing
from src.modules.utils import load_yaml

def main():
    config = load_yaml("C:/Tuan/GitHub/Human-Activity-Recognition/config/data/preprocessing.yaml")

    data = DataPreprocessing(**config)

    data("dataset")


if __name__ == "__main__":
    main()