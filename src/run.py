

# Setup root directory
from rootutils import autosetup
autosetup()

import hydra, shutil
from omegaconf import DictConfig
from src.har import HAR



@hydra.main(config_path="C:/Tuan/GitHub/Human-Activity-Recognition/config/run", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # # Remove the hydra outputs
    # shutil.rmtree("C:/Tuan/GitHub/Human-Activity-Recognition/src/outputs")

    # Load video
    har = HAR(**cfg["har"])

    har.setup_backbone(config= cfg)

    har.run()


if __name__ == "__main__":
    main()