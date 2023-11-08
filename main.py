import cv2
import numpy as np
from ultralytics import YOLO
from Modules.video import VideoProcessing
from Models.detection import HumanDectection
from Models.heatmap import HeatMap



# Loading video and processing video
video = VideoProcessing("C:/Tuan/GitHub/Human-Activity-Recognition/Data/Videos/test.mp4", 4, (500, 500))
data = video()

# Density of heatmap
density = np.zeros((data[0].shape[0], data[0].shape[1]), dtype=np.float32)

# Loading model
model = YOLO("C:/Tuan/GitHub/Human-Activity-Recognition/Models/YOLOv8/yolov8n.pt")
model.fuse()


for frame in data:

    detection = HumanDectection(frame= frame,
                                model= model)

    bounding_boxes = detection()

    heatmap = HeatMap(frame, bounding_boxes, density)

    frame, density = heatmap()

    cv2.imshow("Heatmap", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()