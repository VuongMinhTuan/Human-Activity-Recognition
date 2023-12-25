import math, cv2, time, os, itertools, numpy as np
from functools import cached_property
from typing import Dict, Tuple, Union
from collections import deque
from datetime import datetime
from rich import print
from tqdm import tqdm
from src.modules.utils import tuple_handler
from src.backbone import Backbone


class HAR:
    def __init__(
        self,
        path: str,
        speed: int = 1,
        delay: int = 1,
        subsampling: int = 1,
        sync: bool = True,
        resolution: Tuple = None,
        progress_bar: bool = True,
        show_fps: Union[Dict, bool] = None,
        record: Union[Dict, bool] = None,
    ) -> None:
        
        if not os.path.exists(path):
            raise FileExistsError(
                "File not found. Check again or use an absolute path."
            )
        
        self.path = str(path)
        self.video_capture = cv2.VideoCapture(path)
        self.is_camera = bool(self.total_frame == -1)
        self.__check_speed(speed)
        self.wait = int(delay)
        self.subsampling = max(1, int(subsampling))
        self.sync = bool(sync)
        self.resolution = tuple_handler(resolution, max_dim= 2) if resolution else None
        self.__setup_progress_bar(show=progress_bar)
        self.__setup_fps_display(config=show_fps)
        self.__setup_recorder(config=record)


    def __check_speed(self, value: Union[int, float]) -> None:
        if self.is_camera:
            self.speed = 1
        else:
            self.speed = int(max(1, value))

            if isinstance(value, float):
                self.speed_mul = value / self.speed

    def __setup_progress_bar(self, show: bool) -> None:
        self.progress = tqdm(
            disable= not show,
            total= self.total_frame,
            desc= f"  {self.name}",
            unit= " frame",
            smoothing= 0.3,
            delay= 0.1,
            colour= "cyan",
        )

    def __setup_fps_display(self, config: Union[Dict, bool]) -> None:
        if config not in [False, None]:
            self.fps_history = deque(maxlen= config.get("smoothness", 30))
            self.fps_pos = tuple_handler(config.get("position", (20, 40)), max_dim= 2)


    def __setup_recorder(self, config: Union[Dict, bool]) -> None:
        # Disable if config is not provided
        if not config:
            return

        # Set save folder
        save_folder = os.path.join(
            config["path"],
            datetime.now().strftime("%d-%m-%Y") if self.is_camera else self.stem,
        )

        # Create save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        # Config writer
        save_path = os.path.join(save_folder, config["name"] + ".mp4")

        codec = cv2.VideoWriter_fourcc(*"mp4v")

        fps = float(config["fps"] if config["fps"] else self.fps)

        self.recorder_res = (
            tuple_handler(config["resolution"], max_dim=2)
            if config["resolution"]
            else self.size()
        )

        # Config writer
        self.recorder = cv2.VideoWriter(
            filename=save_path, fourcc=codec, fps=fps, frameSize=self.recorder_res
        )

        # Logging
        print(f"[INFO] [bold]Save recorded video to:[/] [green]{save_path}[/]")


    def __resync(func):
        # Create wrapper function
        def wrapper(self):
            # Check on first run
            if not hasattr(self, "start_time"):
                self.start_time = time.time()

            # Run the function
            output = func(self)

            # Get delay time
            delay = time.time() - self.start_time

            # Check if sync is enable
            if self.sync:
                # Calculate sync value
                sync_time = (
                    1 / self.fps / (self.speed_mul if hasattr(self, "speed_mul") else 1)
                )

                # Apply sync if needed
                if delay < sync_time:
                    time.sleep(sync_time - delay)

            # Display fps if specified
            if hasattr(self, "fps_history"):
                self.fps_history.append(math.ceil(1 / (time.time() - self.start_time)))
                self.add_text(
                    text=f"FPS: {math.ceil(np.mean(self.fps_history))}",
                    pos=self.fps_pos,
                    thickness=2,
                )

            # Setup for new circle
            self.start_time = time.time()

            # Return function output
            return output

        return wrapper

    def __iter__(self) -> "HAR":
        # Video iteration generate
        def generate():
            for _, frame in iter(self.video_capture.read, (False, None)):
                yield frame

        # Generate frame queue
        self.queue = itertools.islice(generate(), 0, None, self.speed)

        # Initialize
        self.pause = False

        # print("[bold]Video progress:[/]")

        return self


    @__resync
    def __next__(self) -> Union[cv2.Mat, np.ndarray]:
        # Get current frame
        self.current_frame = next(self.queue)

        # Change video resolution
        if self.resolution:
            self.current_frame = cv2.resize(self.current_frame, self.resolution)

        # Backbone process
        if hasattr(self, "backbone"):
            # Check subsampling
            if (self.progress.n % self.subsampling) == 0:
                # Process the current frame
                self.backbone.process(self.current_frame)

            # Apply to current frame
            self.current_frame = self.backbone.apply(self.current_frame)

        # Recorder the video
        if hasattr(self, "recorder"):
            self.recorder.write(
                cv2.resize(self.current_frame, self.recorder_res)
                if self.recorder_res
                else self.current_frame
            )

        # Update progress
        self.progress.update(
            max(1, min(self.speed, self.total_frame - self.progress.n))
        )

        # Return current frame
        return self.current_frame

    def __len__(self) -> int:
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


    @cached_property
    def name(self) -> str:
        return self.path.split("/")[-1]


    @cached_property
    def stem(self) -> str:
        return self.name.split(".")[0]


    @cached_property
    def cap(self) -> cv2.VideoCapture:
        return self.video_capture


    @cached_property
    def total_frame(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


    @cached_property
    def fps(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FPS))


    @cached_property
    def shortcuts(self) -> Dict:
        return {
            "quit": "q",
            "pause": "p",
            "resume": "r",
            "detector": "1",
            "classifier": "2",
            "heatmap": "3",
            "track_box": "4"
        }

    def setup_backbone(self, config: Dict) -> None:
        self.backbone = Backbone(
            video= self, process_config= config, **config["backbone"]
        )


    def custom_shortcut(self, values: Dict):
        self.shortcuts.update(
            {name: key for name, key in values.items() if name in self.shortcuts}
        )


    def size(self, reverse: bool = False) -> Tuple[int, int]:
        w, h = (
            int(self.cap.get(prop))
            for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
        )

        return (w, h) if not reverse else (h, w)


    def add_box(
        self,
        top_left: Tuple,
        bottom_right: Tuple,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        
        cv2.rectangle(
            img= self.current_frame,
            pt1= tuple_handler(top_left, max_dim= 2),
            pt2= tuple_handler(bottom_right, max_dim= 2),
            color= tuple_handler(color, max_dim= 3),
            thickness= int(thickness),
        )

    def add_circle(
        self,
        center: Tuple,
        radius: int,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        
        cv2.circle(
            img= self.current_frame,
            center= tuple_handler(center, max_dim=2),
            radius= int(radius),
            color= tuple_handler(color, max_dim=3),
            thickness= int(thickness),
        )

    def add_point(
        self, center: Tuple, radius: int, color: Tuple = (255, 255, 255)
    ) -> None:
        
        cv2.circle(
            img= self.current_frame,
            center= tuple_handler(center, max_dim=2),
            radius= int(radius),
            color= tuple_handler(color, max_dim=3),
            thickness= -1,
        )

    def add_text(
        self,
        text: str,
        pos: Tuple,
        font_scale: int = 1,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        
        cv2.putText(
            img= self.current_frame,
            text= str(text),
            org= tuple_handler(pos, max_dim=2),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
            fontScale= int(font_scale),
            color= tuple_handler(color, max_dim=3),
            thickness= int(thickness),
        )


    def show(self) -> None:
        # Show the frame
        if not hasattr(self, "current_frame"):
            raise ValueError(
                "No current frame to show. Please run or loop through the video first."
            )
        
        cv2.imshow(self.stem, self.current_frame)


    def run(self) -> None:
        # Runs the video playback loop
        for _ in self:
            self.show()

            if not self.delay(self.wait):
                break

        self.release()


    def delay(self, value: int) -> bool:
        key = cv2.waitKey(value if not self.pause else 0) & 0xFF

        # Check pause status
        self.pause = (
            True

            if key == ord(self.shortcuts["pause"])
            else False

            if key == ord(self.shortcuts["resume"])
            else self.pause
        )

        # Check features toggle
        if hasattr(self, "backbone"):
            for process in self.backbone.status:
                if process != "human_count" and key == ord(self.shortcuts[process]):
                    self.backbone.status[process] = not self.backbone.status[process]

        # Check continue
        return True if not key == ord("q") else False

    def release(self) -> None:
        """Release capture"""
        self.video_capture.release()

        if hasattr(self, "recorder"):
            self.recorder.release()

        if hasattr(self, "backbone"):
            self.backbone.finish()

        cv2.destroyWindow(self.stem)