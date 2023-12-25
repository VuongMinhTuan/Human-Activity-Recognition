import os
import torch
from rich import print
from torch.nn import Module
from typing import Dict, Tuple, Union
from src.modules import DataTransformation, device_handler, tuple_handler


CLASSES = ["idle", "laptop", "phone", "walk"]


class Classifier:
    def __init__(
        self,
        checkpoint: str,
        image_size: Union[int, Tuple] = 224,
        half: bool = False,
        optimize: bool = False,
        device: str = "auto",
    ):
        
        # Check if the provided checkpoint path exists
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        

        self.ckpt = checkpoint

        # Determine the device based on user input or availability
        self.device = device_handler(device)

        # Defind transform
        self.transform = DataTransformation.toPIL(
            image_size= tuple_handler(image_size, max_dim=2)
        )

        # Load model
        self.model = torch.load(self.ckpt, map_location= self.device)

        self.half = half

        # Apply half-precision if specified
        if self.half:
            if self.device == "cpu":
                print(
                    "[yellow][WARNING] [Classifier]: Half is only supported on CUDA. Using default float32.[/]"
                )
                
                self.half = False
            else:
                self.model = self.__half(self.model)

        # Apply TorchDynamo compilation if specified
        if optimize:
            self.model = self.__compile(self.model)

        # Store configuration options
        self.model.to(self.device)


    def __call__(self, image: torch.Tensor) -> str:
        return self.forward(image)


    def __half(self, X: Union[torch.Tensor, Module]) -> Union[torch.Tensor, Module]:
        return X.half()


    def __compile(self, X: Module) -> Module:
        return torch.compile(
            model= X,
            fullgraph= True,
            backend= "inductor",
            options= {
                "shape_padding": True,
                "triton.cudagraphs": True,
            }
        )


    def __check_dim(self, X: torch.Tensor) -> torch.Tensor:
        match X.dim():
            case 3:
                X = X.unsqueeze(0)
            case 4:
                pass
            case _:
                raise ValueError(
                    f"Input dimension must be 3 (no batch) or 4 (with batch). Got {X.dim()} instead."
                )
        
        return X


    def forward(self, image: torch.Tensor) -> Dict:
        # Transform
        X = self.transform(image)

        # Get result
        with torch.inference_mode():
            # Check dimension
            X = self.__check_dim(X).to(self.device)

            # Apply haft
            X = self.__half(X) if self.half else X

            outputs = self.model(X)

            outputs = torch.softmax(outputs, dim=1)

            value, pos = torch.max(outputs, dim=1)

        return {"label": CLASSES[pos.item()], "score": value.item()}