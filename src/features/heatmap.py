import cv2, numpy as np
from typing import Tuple
from pathlib import Path
from src.modules.utils import tuple_handler


class HeatMap:
    def __init__(
        self,
        shape: Tuple[int, int],
        grow_value: int = 3,
        decay_value: int = 1,
        blurriness: float = 1.0,
    ):
        
        self.layer = np.zeros(shape= tuple_handler(shape, max_dim= 2), dtype= np.uint8)
        self.grow_value = grow_value
        self.decay_value = decay_value
        self.blurriness = blurriness


    def check(self, area: Tuple) -> None:
        # Grow
        x1, y1, x2, y2 = tuple_handler(area, max_dim=4)
        x1, y1 = int(x1 * 0.95), int(y1 * 0.95)
        x2, y2 = int(x2 * 1.05), int(y2 * 1.05)

        self.layer[y1:y2, x1:x2] = np.minimum(
            self.layer[y1:y2, x1:x2] + self.grow_value, 255 - self.grow_value
        )

    def update(self) -> None:
        # Update the heatmap

        # Decay
        self.layer = ((1 - self.decay_value / 100) * self.layer).astype(np.uint8)

        # Blur
        blurriness = int(self.blurriness * 100)
        blurriness = blurriness + 1 if blurriness % 2 == 0 else blurriness
        self.layer = cv2.stackBlur(self.layer, (blurriness, blurriness), 0)

        self.heatmap = cv2.applyColorMap(self.layer, cv2.COLORMAP_TURBO)

        # Check if save video
        if hasattr(self, "_video_writer"):
            self._video_writer.write(self.heatmap)

        # Check if save image
        if hasattr(self, "_image_writer"):
            self._image_writer["count"] += 1
            self._image_writer["image"] += self.heatmap
            cv2.imwrite(
                self._image_writer["path"],
                (self._image_writer["image"] / self._image_writer["count"]).astype(
                    np.uint8
                ),
            )


    def get(self) -> np.ndarray:
        return self.heatmap


    def save_video(
        self, save_path: str, fps: int, size: Tuple, codec: str = "mp4v"
    ) -> None:
        
        save_path = Path(save_path)

        # Create save folder
        save_path.parent.mkdir(parents= True, exist_ok= True)

        # Create video writer
        self._video_writer = cv2.VideoWriter(
            filename= str(save_path),
            fourcc= cv2.VideoWriter_fourcc(*codec),
            fps= fps,
            frameSize= size,
        )


    def save_image(self, save_path: str, size: Tuple) -> None:
        #  Create save folder
        Path(save_path).parent.mkdir(parents= True, exist_ok= True)

        self._image_writer = {
            "path": save_path,
            "count": 0,
            "image": np.zeros(shape= (*size, 3), dtype= np.float32),
        }


    def release(self):
        # Release capture
        if hasattr(self, "writer"):
            self.writer.release()