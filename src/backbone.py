import os, cv2, threading, numpy as np
from functools import cached_property
from typing import Dict, Union
from datetime import datetime
from copy import deepcopy
from queue import Queue
from rich import print
from cv2 import Mat
from src.features import HeatMap
from src.modules import tuple_handler
from src.tasks import Detector, Classifier


class Backbone:
    def __init__(
        self,
        video: "HAR",
        mask: bool = False,
        thread: bool = False,
        background: bool = False,
        save: Union[Dict, bool] = None,
        process_config: Union[Dict, bool] = None,
    ) -> None:
        
        self.video = video
        self.mask = mask
        self.thread = thread
        self.background = background
        self.__setup_save(config=save)
        self.queue = Queue()

        # Process status:
        #   True by default
        self.status = {"detector": True, "human_count": True}
        
        #   False by default
        self.status.update(
            {
                process: False
                for process in [
                    "classifier",
                    "heatmap",
                    "track_box",
                ]
            }
        )

        # Setup each process
        for process in self.status:
            if process_config.get(process, False) or process_config["features"].get(
                process, False
            ):
                args = (
                    [process_config["features"][process]]
                    if process not in ["detector", "classifier"]
                    else [process_config[process], process_config["device"]]
                )
                getattr(self, f"_setup_{process}")(*args)


    def __call__(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        return self.process(frame)


    def __setup_save(self, config: Union[Dict, bool]) -> None:
        # Disable if config is not provided
        if not config:
            return

        # Set save path
        self.save_path = os.path.join(
            config["path"],
            datetime.now().strftime("%d-%m-%Y")
            if self.video.is_camera
            else self.video.stem,
        )

        # Create destination folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok= True)

        # Set save frequency
        self.save_interval = config["interval"]

        # Logging
        print(f"[INFO] [bold]Save process result to:[/] [green]{self.save_path}[/]")


    def _setup_detector(self, config: Dict, device: str) -> None:
        self.detector = Detector(**config["model"], device= device)
        self.show_detected = config["show"]
        self.track = config["model"]["track"]


    def _setup_classifier(self, config: Dict, device: str) -> None:
        self.classifier = Classifier(**config["model"], device= device)
        self.show_classified = config["show"]


    def _setup_heatmap(self, config: Dict) -> None:
        self.heatmap = HeatMap(shape=self.video.size(reverse= True), **config["layer"])
        self.heatmap_opacity = config["opacity"]

        if hasattr(self, "save_path") and config["save"]:
            # Save video
            if config["save"]["video"]:
                self.heatmap.save_video(
                    save_path= os.path.join(self.save_path, "heatmap.mp4"),
                    fps= self.video.fps / self.video.subsampling,
                    size= self.video.size(),
                )
    
            # Save image
            if config["save"]["image"]:
                self.heatmap.save_image(
                    save_path=os.path.join(self.save_path, "heatmap.jpg"),
                    size=self.video.size(reverse= True),
                )


    # def _setup_human_count(self, config: Dict) -> None:
    #     self.human_count = HumanCount(smoothness= config["smoothness"])
    #     self.human_count_position = config["position"]

    #     if hasattr(self, "save_path") and config["save"]:
    #         self.human_count.config_save(
    #             save_path= os.path.join(self.save_path, "human_count.csv"),
    #             interval= self.save_interval,
    #             fps= int(self.video.fps / self.video.subsampling),
    #             speed= self.video.speed,
    #             camera= self.video.is_camera,
    #         )


    # def _setup_track_box(self, config: Dict) -> None:
    #     self.track_box = TrackBox(
    #         default_config= config["default"], boxes=config["boxes"]
    #     )

    #     if hasattr(self, "save_path") and config["save"]:
    #         self.track_box.config_save(
    #             save_path= os.path.join(self.save_path, "track_box.csv"),
    #             interval= self.save_interval,
    #             fps= int(self.video.fps / self.video.subsampling),
    #             speed= self.video.speed,
    #             camera= self.video.is_camera,
    #         )


    @cached_property
    def __new_mask(self) -> np.ndarray:
        return np.zeros((*self.video.size(reverse= True), 3), dtype= np.uint8)


    def __process_is_activate(self, name: str, background: bool = False) -> bool:
        # Check if a process is activate
        return hasattr(self, name) and (
            self.status[name] or (self.background if background else False)
        )


    def __threaded_process(func):
        # Move process to a separate thread

        def wrapper(self, frame):
            # Check if using Thread
            if not self.thread:
                return func(self, frame)

            # Turn mask on when using Thread
            elif not self.mask:
                self.mask = True

            # Only spawn Thread on first run or Thread is free
            if (
                not hasattr(self, "current_process")
                or not self.current_process.is_alive()
            ):
                # Spam a new thread
                self.current_process = threading.Thread(
                    target=func, args=(self, frame), daemon= True
                )

                # Start running the thread
                self.current_process.start()

        return wrapper


    @__threaded_process
    def process(self, frame: Union[np.ndarray, Mat]) -> None:
        # Check mask option
        mask = deepcopy(self.__new_mask if self.mask else frame)

        # Skip all of the process if detector is not specified
        if self.__process_is_activate("detector", background= True):
            # Get detector output
            boxes = self.detector(frame)

            # Lambda function for dynamic color apply
            dynamic_color = lambda x: (0, x * 400, ((1 - x) * 400))

            # Human count
            if hasattr(self, "human_count"):
                # Update new value
                self.human_count.update(value=len(boxes))

                # Add to frame
                cv2.putText(
                    img= mask,
                    text= f"Person: {self.human_count.get_value()}",
                    org= self.human_count_position,
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale= 1,
                    color= tuple_handler(255, max_dim=3),
                    thickness= 2,
                )

            # Loop through the boxes
            for detect_output in boxes:
                # xyxy location
                x1, y1, x2, y2 = detect_output["box"]

                # Center point
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Check detector show options
                if self.__process_is_activate("detector") and self.show_detected:
                    # Apply dynamic color
                    color = (
                        dynamic_color(detect_output["score"])
                        if self.show_detected["dynamic_color"]
                        else 255
                    )

                    # Show dot
                    if self.show_detected["dot"]:
                        cv2.circle(
                            img= mask,
                            center= center,
                            radius= 5,
                            color= color,
                            thickness= -1,
                        )

                    # Show box
                    if self.show_detected["box"]:
                        cv2.rectangle(
                            img= mask,
                            pt1= (x1, y1),
                            pt2= (x2, y2),
                            color= color,
                            thickness= 2,
                        )

                    # Show score
                    if self.show_detected["score"]:
                        cv2.putText(
                            img= mask,
                            text= f"{detect_output['score']:.2}",
                            org= (x1, y2 - 5),
                            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 1,
                            color= color,
                            thickness= 2,
                        )

                # Show id it track
                if self.track:
                    cv2.putText(
                        img= mask,
                        text= detect_output["id"],
                        org= (x1, y1 - 5),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale= 1,
                        color= tuple_handler(255, max_dim= 2),
                        thickness= 2,
                    )

                # Classification
                if (
                    self.__process_is_activate("detector")
                    and self.__process_is_activate("classifier")
                    and self.show_classified
                ):
                    # Add box margin
                    box_margin = 10
                    human_box = frame[
                        max(0, y1 - box_margin) : min(frame.shape[1], y2 + box_margin),
                        max(0, x1 - box_margin) : min(frame.shape[1], x2 + box_margin),
                    ]

                    # Get model output
                    classify_output = self.classifier(human_box)

                    # Format result
                    classify_result = ""
                    if self.show_classified["text"]:
                        classify_result += classify_output["label"]

                    if self.show_classified["score"]:
                        classify_result += f' ({classify_output["score"]:.2})'

                    # Add to frame, color based on score
                    cv2.putText(
                        img= mask,
                        text= classify_result,
                        org= (x1, y1 - 5),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale= 1,

                        color= (
                            dynamic_color(classify_output["score"])
                            if self.show_classified["dynamic_color"]
                            else 255
                        ),

                        thickness=2,
                    )

                # Update heatmap
                if self.__process_is_activate("heatmap", background= True):
                    self.heatmap.check(area= (x1, y1, x2, y2))

                # Check for track box
                if self.__process_is_activate("track_box"):
                    self.track_box.check(pos= center)

            # Apply heatmap
            if self.__process_is_activate("heatmap", background= True):
                self.heatmap.update()

            # Add track box to frame
            if hasattr(self, "track_box") and self.status["track_box"]:
                self.track_box.update()
                self.track_box.apply(mask)

        # Put result to a safe thread
        self.queue.put(mask)


    def apply(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        # Check if any processes are completed
        if not self.queue.empty():
            self.overlay = self.queue.get()
            self.filter = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY) != 0

        # Return on result is empty and not mask
        elif not self.mask:
            return frame

        # Enable heatmap
        if self.__process_is_activate("heatmap") and hasattr(self.heatmap, "heatmap"):
            cv2.addWeighted(
                src1= self.heatmap.get(),
                alpha= self.heatmap_opacity,
                src2= frame if self.mask else self.overlay,
                beta= 1 - self.heatmap_opacity,
                gamma= 0,
                dst= frame if self.mask else self.overlay,
            )

        # Return overlay when not using mask
        if not self.mask:
            return self.overlay

        # Check if first run
        if hasattr(self, "overlay"):
            frame[self.filter] = self.overlay[self.filter]

        return frame


    def finish(self) -> None:
        if hasattr(self, "heatmap"):
            self.heatmap.release()