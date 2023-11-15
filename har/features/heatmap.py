import numpy as np
import cv2





class HeatMap:
    def __init__(self, frame: np.ndarray, bounding_boxes: np.ndarray, density: np.ndarray):
        self.frame = frame
        self.bounding_boxes = bounding_boxes
        self.denstity = density


    def __call__(self):
        self.denstity = self.create_ellipse(self.bounding_boxes, self.denstity)
        self.frame = self.create_heatmap(self.frame, self.denstity)

        return (self.frame, self.denstity)


    def create_ellipse(self, bounding_boxes: np.ndarray, density: np.ndarray):
        for box in bounding_boxes:
            x, y, w, h = (int(b.item()) for b in box)

            center = (x, y)
            size = (int(w/2 * 0.04), int(h/2 * 0.04))
            angle = 0
            startAngle = 0
            endAngle = 360

            cv2.ellipse(density, center, size, angle, startAngle, endAngle, (1, 1), -1)

        return density
    

    def create_heatmap(self, frame: np.ndarray, density: np.ndarray):
        density = cv2.stackBlur(density, (21, 21), 0)
        density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(density, cv2.COLORMAP_TURBO)
        frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        return frame