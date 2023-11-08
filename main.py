import cv2
import numpy as np
from Modules.video import VideoProcessing
from Models.detection import HumanDectection
from Models.heatmap import HeatMap




video = VideoProcessing("C:/Tuan/GitHub/Human-Activity-Recognition/Data/Videos/test.mp4", 4, (500, 500))
data = video()
density = np.zeros((data[0].shape[0], data[0].shape[1]), dtype=np.float32)


for frame in data:

    detection = HumanDectection(source= "C:/Tuan/GitHub/Human-Activity-Recognition/Models/YOLOv8/yolov8n.pt",
                                frame= frame)

    bounding_boxes = detection()

    heatmap = HeatMap(frame, bounding_boxes, density)

    frame, density = heatmap()

    cv2.imshow("Heatmap", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()