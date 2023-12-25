import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Union
from functools import partial
from torch.nn import Module
from rich import print
from ultralytics import YOLO
from src.modules.utils import device_handler


class Detector:
    def __init__(
        self,
        weight: str = "C:/Tuan/GitHub/Human-Activity-Recognition/weights/yolov8n.pt",
        track: bool = False,
        conf: float = 0.25,
        iou: float = 0.7,
        size: Union[int, Tuple] = 640,
        half: bool = False,
        fuse: bool = False,
        onnx: bool = False,
        optimize: bool = False,
        backend: str = None,
        device: str = "auto",
        **kwargs
    ):
       
        self.device = device_handler(device)

        self.model = self.__get_model(
            weight= weight,
            track= track,
            fuse= fuse,
            format= "onnx" if onnx else "pt",
            optimize= optimize,
            backend= backend,

            config= {
                "conf": conf,
                "iou": iou,
                "imgsz": size,
                "half": self.__check_half(half),
                "device": self.device,
            },

            **kwargs
        )


    def __call__(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        return self.forward(image)
    

    def __compile(self, X: Module, backend: str) -> Module:
        # Determine the backend to use for compilation
        backend = (
            "inductor"
            if not backend
            or backend not in torch._dynamo.list_backends()
            or (backend == "onnxrt" and not torch.onnx.is_onnxrt_backend_supported())
            else backend
        )

        # Compile the model using the specified backend and additional options
        return torch.compile(
            model= X,
            fullgraph= True,
            backend= backend,
            options= {
                "shape_padding": True,
                "triton.cudagraphs": True,
            }
        )


    def __get_model(
        self,
        weight: str,
        track: bool,
        fuse: bool,
        optimize: bool,
        backend: str,
        config: Dict,
        **kwargs
    ):
        
        # Create an instance of the YOLO model
        model = YOLO(weight, task= "detect")

        # Fuse model layers if specified
        if fuse:
            model.fuse()

        # Optimize the model using torch.compile if specified
        if optimize:
            model = self.__compile(X= model, backend= backend)

        # Configure the track model if specified
        if track:
            model = model.track

            config.update(
                {
                    "persist": True,
                    "tracker": "C:/Tuan/GitHub/Human-Activity-Recognition/config/trackers/tracker.yaml"
                }
            )
        else:
            model = model.predict

        # Return a partially configured YOLO model
        return partial(model, **config, classes= 0, verbose= False, **kwargs)


    def __check_half(self, half: bool) -> bool:
        # Check if half precision is specified and the device is CPU
        if half and self.device == "cpu":
            print(
                "[yellow][WARNING] [YOLOv8]: Half is only supported on CUDA. Using default float32.[/]"
            )

            half = False

        return half


    def forward(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        # Perform a forward pass of the model on the input image
        result = self.model(source= image)[0]

        outputs = []

        # Extract information from the detection results
        if result.boxes:
            for box in result.boxes:
                outputs.append(
                    {
                        "id": int(box.id.item()) if box.is_track else None,
                        "box": [int(i.item()) for i in box.xyxy[0]],
                        "score": box.conf.item()
                    }
                )

        return outputs