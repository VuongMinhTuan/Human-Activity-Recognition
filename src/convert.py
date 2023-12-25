# Setup root directory
from rootutils import autosetup
autosetup()

import os, torch
from src.modules import LitModule, load_yaml
from src.models import VIT
from typing import Dict



def main(cfg: Dict):
    # Define model
    MODEL = VIT(
        num_classes= len(os.listdir(os.path.join(cfg["data"]["dataset_dir"], "train"))),
        **cfg["model"]
    )


    # Module
    LIT_MODULE = LitModule(
        model= MODEL,
        checkpoint= "C:/Tuan/GitHub/Human-Activity-Recognition/logs/version_1/checkpoints/epoch=55-step=117152.ckpt"
    )


    # Save model to .pt file
    torch.save(LIT_MODULE, "vit.pt")



if __name__ == "__main__":
    # Convert .ckpt file to .pt file
    main(load_yaml("C:/Tuan/GitHub/Human-Activity-Recognition/config/classifiers/train.yaml"))