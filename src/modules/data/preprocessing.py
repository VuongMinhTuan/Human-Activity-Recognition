import cv2, os, splitfolders, psutil, numpy as np
from typing import List, Tuple, Union
from pathlib import Path
from PIL import Image
from tqdm.contrib.concurrent import process_map





class ImageProcessing:
    # Resize image to the specified dimensions
    @staticmethod
    def resize(
        image: np.ndarray,
        size: Union[int, List[int], Tuple[int]]
    ) -> np.ndarray:
        
        return cv2.resize(image, size)
    

    # Adds a border around a given video frame to make it a square frame
    @staticmethod
    def add_border(
        image: np.ndarray, border_color: Tuple | int = (0, 0, 0)
    ) -> np.ndarray:
       
        img_h, img_w = image.shape[:2]
        target_size = max(img_h, img_w)

        border_v = (target_size - img_h) // 2
        border_h = (target_size - img_w) // 2

        return cv2.copyMakeBorder(
            image,
            border_v,
            border_v,
            border_h,
            border_h,
            cv2.BORDER_CONSTANT,
            border_color,
        )



class VideoProcessing:
    # Load dataset and extract it to frames
    @staticmethod
    def load(path: str):
        # Check directory if it existed
        if not os.path.exists(path):
            raise FileExistsError("File is not found!!!")
        
        # Load video
        video = cv2.VideoCapture(path)

        # Check video
        if not video.isOpened():
            raise RuntimeError("Could not open the video file!!!")
        

        # Extract frames from video
        output = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _, frame in iter(video.read, (False, None))
        ]

        video.release()

        return output
    

    # Do sampling on video
    @staticmethod
    def sampling(video: Union[np.ndarray, List[np.ndarray]], value: int):
        return video[::value]
    

    # Truncate video to specificed maximum number of frames
    @staticmethod
    def truncating(video: Union[np.ndarray, List[np.ndarray]], max_frame: int):
        middle_frame = len(video) // 2
        m = max_frame // 2
        r = max_frame % 2

        return video[middle_frame - m : middle_frame + m + r]
    

    # Pad video with black frames to meet a minimum frame length
    @staticmethod
    def padding(video: Union[np.ndarray, List[np.ndarray]], min_frame: int):
        zeros_array = np.zeros((min_frame, *np.array(video).shape[1:]), dtype= np.uint8)
        zeros_array[: len(video), ...] = video

        return zeros_array
    

    #  Resize each frame of video to the specified dimensions
    @staticmethod
    def resize(video: Union[np.ndarray, List[np.ndarray]], size: Union[int, List[int], Tuple[int]]):
        return [cv2.resize(frame, size) for frame in video]
    




class DataPreprocessing:
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.classes = os.listdir(self.cfg['dataset_dir'])
        self.extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".mpg"]


    def __call__(self, save_folder: str= None):
        return self.auto(save_folder)
        

    # Split dataset into train, validation and test dataset
    def split_data(self, input_folder: str, output_folder: str):
        # Check directory if it existed
        if not os.path.exists(input_folder):
            raise FileExistsError("File is not found.")
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)


        # Find the number of validation and test files in dataset
        fixed = tuple(int(min(
            [os.listdir(Path.joinpath(self.cfg['dataset_dir'], cls)).__len__() for cls in self.classes]
        ) * self.cfg['ratio'][1:]))


        splitfolders.fixed(
            input= input_folder,
            output= output_folder,
            seed= 1337,
            fixed= fixed,
            oversample= False,
            group_prefix= None,
            move= False
        )


    # Process dataset
    def __process_data(self, path: str) -> List[np.ndarray]:
        video = VideoProcessing.load(path)

        if self.cfg['sampling'] != 0:
            video = VideoProcessing.sampling(np.array(video), self.cfg['sampling'])
        
        if self.cfg['max_frame'] != 0:
            video = VideoProcessing.truncating(np.array(video), self.cfg['max_frame'])
        
        if self.cfg['min_frame'] != 0:
            video = VideoProcessing.padding(np.array(video), self.cfg['min_frame'])
        
        if self.cfg['image_size']:
            video = VideoProcessing.resize(np.array(video), self.cfg['image_size'])
        
        return video
    

    # Generate frames from video
    def __generate_frame(self, path: str):
        # Check directory if it existed
        if not os.path.exists(path):
            raise FileExistsError("File is not found.")
        
        # Get file name
        file_name = Path(path).stem

        # Create destination path
        dst_path = Path(self.cfg["save_dir"])

        # Process dataset
        video = self.__process_data(path)

        for i, frame in enumerate(video):
            save_path = os.path.join(dst_path, f"{file_name}_{i}.jpg")

            if not os.path.exists(save_path):
                image = Image.fromarray(frame)
                image.save(save_path)


    def auto(self, save_folder: str= None):
        # Create folder for images of dataset
        if save_folder is None:
            self.cfg["save_dir"] = Path(self.cfg["save_dir"]).joinpath(self.cfg['folder_name'] + "_images").as_posix()
            os.makedirs(self.cfg["save_dir"], exist_ok= True)
        else:
            self.cfg["save_dir"] = Path(self.cfg["save_dir"]).joinpath(save_folder).as_posix()
            os.makedirs(self.cfg["save_dir"], exist_ok= True)


        # Process summary
        print(f"\n[bold]Summary:[/]")
        print(f"  Number of workers: {self.cfg['num_workers']}")
        print(f"  Data path: [green]{self.cfg['dataset_dir']}[/]")
        print(f"  Save path: [green]{self.cfg['save_dir']}[/]")

        # Calcute chunksize base on cpu parallel power
        benchmark = lambda x: max(
            1, round(len(x) / (self.cfg['num_workers'] * psutil.cpu_freq().max / 1000) / 4)
        )

        # Generate data
        print("\n[bold][yellow]Generating data...[/][/]")

        video_paths = [
            str(video)
            for ext in self.extensions
            for video in Path(self.cfg['dataset_dir']).rglob("*" + ext)
        ]

        process_map(
            self.__generate_frame,
            video_paths,
            max_workers= self.cfg['num_workers'],
            chunksize= benchmark(video_paths),
        )

        print("\n[bold][green]Processing data complete.[/][/]")