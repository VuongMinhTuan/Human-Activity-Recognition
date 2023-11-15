import torch
from yolo import Detector




class Tracker(Detector):
    def __init__(self, source):
        super().__init__(source)


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
        
        results = self._model.track(
            source = frame,
            classes = classes,
            conf = conf,
            iou = iou,
            size = imgsz,
            half = half,
            device = device,
            tracker = tracker, 
            persist = persist
        )

        self._bounding_boxes = results[0].boxes.xyxy
        self._conf = results[0].boxes.conf

        return results
    

    # Annotate frame with bounding_boxes
    def annotate(self, results):
        return results[0].plot()