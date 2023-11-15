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
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"


        self.track = track
        self.model = weight if weight else "C:/Tuan/GitHub/Human-Activity-Recognition/har/yolo/pretrained/yolov8n.pt"

        if device == "cpu":
            half = False

        self.args_map = {"classes": classes, "conf": conf, "iou": iou, "imgsz": imgsz, "half": half, "device": device}


    def __call__(self, frame):
        return self.forward(frame)


    def forward(self, frame):
        
        # Load the model
        model = Tracker(self.model) if self.track else Detector(self.model)

        # Predict the results
        results = model.predict

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
        results = model.annotate(results)


        # Results of detection
        ouputs = []

        bounding_boxes = model.get_bounding_boxes()
        conf = model.get_conf()
        bounding_boxes_xywh = model.xyxy_to_xywh(bounding_boxes)

        for i in range(len(bounding_boxes)):
            human = {"box": [int(box) for box in bounding_boxes[i].tolist()],
                     "box_xywh": [int(box) for box in bounding_boxes_xywh[i].tolist()],
                     "conf": conf[i]}

            ouputs.append(human)


        return results if self.track else ouputs
    



import cv2

video = cv2.VideoCapture("C:/Tuan/GitHub/Human-Activity-Recognition/data/video/test.mp4")

yolo = YOLOv8(track= True)

while video.isOpened():
    success, frame = video.read()

    if success:
        results = yolo(frame)

        cv2.imshow("YOLOv8 tracking", results)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()