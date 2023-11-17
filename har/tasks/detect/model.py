import torch
from detectors import Detector
from trackers import Tracker
from typing import Tuple
from functools import partial


class YOLOv8:
    def __init__(
        self,
        weight: str = None,
        classes: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | Tuple = 640,
        half: bool = False,
        track: bool = False,
        device: str = "auto",
    ):
        
        # Set device for model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            half = False


        # Tracker or Detector
        self.track = track

        # Load the model
        source = weight if weight else "C:/Tuan/GitHub/Human-Activity-Recognition/har/pretrained/yolov8n.pt"
        self.model = Tracker(source) if self.track else Detector(source)

        # Create arguments map
        self.args_map = {"classes": classes, "conf": conf, "iou": iou, "imgsz": imgsz, "half": half, "device": device}


    def __call__(self, frame):
        return self.forward(frame)


    def forward(self, frame):

        # Predict the results
        results = self.model.predict

        # Pass the arguments to model
        results = partial(results, frame, **self.args_map)

        results = (
            results(
                persist= True,
                tracker = "C:/Tuan/GitHub/Human-Activity-Recognition/config/trackers/botsort.yaml"
            )
            if self.track
            else results()
        )


        # Results of tracking
        results = self.model.annotate(results)


        # Results of detection
        ouputs = []

        bounding_boxes = self.model.get_bounding_boxes()
        conf = self.model.get_conf()
        bounding_boxes_xywh = self.model.xyxy_to_xywh(bounding_boxes)

        for i in range(len(bounding_boxes)):
            human = {"box": [int(box) for box in bounding_boxes[i].tolist()],
                     "box_xywh": [int(box) for box in bounding_boxes_xywh[i].tolist()],
                     "conf": conf[i]}

            ouputs.append(human)


        return results if self.track else ouputs