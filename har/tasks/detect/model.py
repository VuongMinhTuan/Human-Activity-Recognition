import torch
from yolo import Tracker, Detector
from typing import Tuple
from functools import partial


class YOLOv8:
    def __init__(
        self,
        weight: str = None,
        classes: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        size: int | Tuple = 640,
        half: bool = False,
        track: bool = False,
        device: str = "auto",
    ):
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.track = track
        self.model = weight if weight else "C:/Tuan/GitHub/Human-Activity-Recognition/har/yolo/pretrained/yolov8n.pt"

        if device == "cpu":
            half = False

        self.args_map = {"classes": classes, "conf": conf, "iou": iou, "imgsz": size, "half": half}


    def __call__(self, frame):
        self.forward(frame)


    def forward(self, frame):
        
        # Load the model
        model = Tracker(self.model).predict if self.track else Detector(self.model).predict

        # Pass the arguments to model
        model = partial(model, frame, **self.args_map)

        model = (
            model(
                persist= True,
                tracker = "C:/Tuan/GitHub/Human-Activity-Recognition/config/trackers/botsort.yaml"
            )
            if self.track
            else model()
        )


        ouputs = []

        bounding_boxes = model.get_bounding_boxes()
        conf = model.get_conf()
        id = model.get_id()

        for i in range(len(bounding_boxes)):
            x1, x2, x3, x4 = [int(box) for box in bounding_boxes[i].tolist()]
            
            human = {"box": (x1, x2, x3, x4)}

            human.update({"Id": id[i], "conf": conf[i]} if self.track else {"conf": conf[i]})

            ouputs.append(human)

        return ouputs
    



    