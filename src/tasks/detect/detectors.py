import torch
from ultralytics import YOLO



class Detector:
    def __init__(self, source):
        self._model = YOLO(source)
        self._bounding_boxes = torch.Tensor()
        self._conf = torch.Tensor()
        self._id = torch.Tensor()


    # Predict objects
    def predict(
        self, 
        frame, 
        classes,
        conf,
        iou,
        imgsz,
        half, 
        device
    ):
        
        results = self._model.predict(
            source = frame,
            classes = classes,
            conf = conf,
            iou = iou,
            imgsz = imgsz,
            half = half,
            device = device
        )

        self._bounding_boxes = results[0].boxes.xyxy
        self._conf = results[0].boxes.conf
        self._id = results[0].boxes.id

        return results
    

    # Get the bounding boxes with xyxy format
    def get_bounding_boxes(self):
        return self._bounding_boxes
    

    # Get the confidence scores of detected objects
    def get_conf(self):
        return self._conf
    

    # Get the id of detected objects
    def get_id(self):
        return self._id
    

    # Convert the bounding boxes from xyxy format to xywh format
    def xyxy_to_xywh(self, xyxy):
        bounding_boxes_xywh = []

        for box in xyxy:
            x1, y1, x2, y2 = (int(b.item()) for b in box)

            w, h = x2 - x1, y2 - y1
            x, y = w/2, h/2

            bounding_boxes_xywh.append([x, y, w, h])

        return torch.Tensor(bounding_boxes_xywh)