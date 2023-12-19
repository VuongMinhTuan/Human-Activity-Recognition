import cv2, numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from typing import Union, Any



class Detector:
    def __init__(
        self,
        weight: str,
        task: Any | None = None
    ):
        # Task of model
        self.task = task
          
        # Get the model
        self.model = YOLO(f"C:/Tuan/GitHub/Human-Activity-Recognition/weights/{weight}", task) if weight.count('/') == 0 else YOLO(weight, task)


    
    def __call__(
        self,
        source: Any | None = None,
        stream: bool = False,
        predictor: Any | None = None,
        persist: bool = False,
        **kwargs: Any
    ):
        
        if self.task != "track" and self.task != "detect":
            LOGGER.warning(f"WARNING ⚠️ {self.task} is invalid!!! Only 'track' and 'detect are supported'")


        if self.task == "track" or self.task is None:
            results = self.predict(source= source, stream= stream, predictor= predictor, **kwargs)
        else:
            results = self.track(source= source, stream= stream, persist= persist, **kwargs)

        return results


    # Predictor mode
    def predict(
        self,
        source: Any | None = None,
        stream: bool = False,
        predictor: Any | None = None,
        **kwargs: Any
    ):
        
        if source is None:
            LOGGER.warning("WARNING ⚠️ 'source' is missing!!!")


        results = self.model.predict(source= source, stream= stream, predictor= predictor, **kwargs)[0]

        outputs = []

        for box in results.boxes:
            outputs.append(
                {
                    "id": int(box.id.item()) if box.is_track else None,
                    "box": (int(i.item()) for i in box.xyxy[0]),
                    "conf_score": box.conf.item()
                }
            )

        return outputs


    # Tracker mode
    def track(
        self,
        source: str = None | Union[cv2.Mat, np.ndarray],
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any
    ):
        
        if source is None:
            LOGGER.warning("WARNING ⚠️ 'source' is missing!!!")


        results = self.model.track(source= source, stream= stream, persist= persist, **kwargs)[0]

        outputs = []

        for box in results.boxes:
            outputs.append(
                {
                    "id": int(box.id.item()) if box.is_track else None,
                    "box": (int(i.item()) for i in box.xyxy[0]),
                    "conf_score": box.conf.item()
                }
            )

        return outputs