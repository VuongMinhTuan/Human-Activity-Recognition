import numpy as np
from ultralytics import YOLO
import cv2


class HumanDectection:
    def __init__(self, source: str, frame: np.ndarray):
        self.frame = frame
        self.path = source


    def __call__(self):
        model = self.load_model(self.path)
        results = self.predict(model, self.frame)

        return results


    def load_model(self, path):
        model = YOLO(path)
        model.fuse()

        return model
    

    def predict(self, model, frame: np.ndarray) -> np.ndarray:
        # Value of Class: 0 = Human
        results = model(frame, classes= 0)
        results = results[0].boxes.xyxy

        return results
    

    def crop_frames(self, frame: np.ndarray, results: np.ndarray):
        boxes = results[0].boxes.xyxy
        human_frames = []

        for box in boxes:
            x1, y1, x2, y2 = (int(b.item()) for b in box)
            human_frames.append(frame[y1:y2, x1: x2])

        return human_frames
    

    def add_border(self, results: np.ndarray):
        frames_bordered = []

        for frame in results:
            frame_h, frame_w = frame.shape[:2]
            target_size = max(frame_h, frame_w)

            border_v = (target_size - frame_h) // 2
            border_h = (target_size - frame_w) // 2

            frames_bordered.append(cv2.copyMakeBorder(frame, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0))


        return frames_bordered
    





# data = cv2.VideoCapture("C:/Tuan/GitHub/Human-Activity-Recognition/Data/Videos/test.mp4")
# frame = data.read()[1]


# detection = HumanDectection(source= "C:/Tuan/GitHub/Human-Activity-Recognition/Models/YOLOv8/yolov8x6.pt",
#                             frame= frame)

# results = detection()
# human_frames = detection.crop_frames(frame, results)
# human_frames = detection.add_border(human_frames)



# for f in human_frames:
#     f = cv2.resize(f, (500, 500))
#     cv2.imshow("X", f)
#     cv2.waitKey(1000)

# cv2.destroyAllWindows()