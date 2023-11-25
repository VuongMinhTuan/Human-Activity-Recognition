import torch
from ultralytics import YOLO




class Tracker:
    def __init__(self, source):
        self.model = YOLO(source)
        self.bounding_boxes = torch.Tensor()
        self.conf = torch.Tensor()
        self.id = torch.Tensor()


    # Override the predict method
    def predict(
        self,
        frame,
        classes,
        conf,
        iou,
        imgsz,
        half, 
        device,
        tracker = None,
        persist = True
    ):
        
        results = self.model.track(
            source = frame,
            classes = classes,
            conf = conf,
            iou = iou,
            imgsz = imgsz,
            half = half,
            device = device,
            tracker = tracker, 
            persist = persist
        )

        self.bounding_boxes = results[0].boxes.xyxy
        self.conf = results[0].boxes.conf
        self.id = results[0].boxes.id

        return results
    

    # Annotate frame with bounding_boxes
    def annotate(self, results):
        return results[0].plot()


    # Get the bounding boxes with xyxy format
    def get_bounding_boxes(self):
        return self.bounding_boxes
    

    # Get the confidence scores of detected objects
    def get_conf(self):
        return self.conf
    

    # Get the id of detected objects
    def get_id(self):
        return self.id
    

    # Convert the bounding boxes from xyxy format to xywh format
    def xyxy_to_xywh(self, xyxy):
        bounding_boxes_xywh = []

        for box in xyxy:
            x1, y1, x2, y2 = (int(b.item()) for b in box)

            w, h = x2 - x1, y2 - y1
            x, y = w/2, h/2

            bounding_boxes_xywh.append([x, y, w, h])

        return torch.Tensor(bounding_boxes_xywh)