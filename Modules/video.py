import os
import numpy as np
import cv2
from typing import Tuple, List



class VideoProcessing:
    def __init__(self, loading_path: str, sampling_value: int, video_size: Tuple[int, int]):
        self.loading_path = loading_path
        self.value = sampling_value
        self.size = video_size


    def __call__(self):
        video = self.__loading(self.loading_path)
        video = self.__sampling(video, self.value)
        video = self.__resize(video, self.size)

        return video

    def __loading(self, path: str) -> List[np.ndarray]:
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        
        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise RuntimeError("Could not open the video file")
        
        output = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, frame in iter(video.read, (False, None))
        ]

        video.release()

        return output


    def __sampling(self, video: np.ndarray, value: int) -> np.ndarray:
        return video[::value] if value else video


    def __resize(self, video: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return np.array([cv2.resize(frame, size) for frame in video])