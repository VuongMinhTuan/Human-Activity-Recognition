import cv2, os, splitfolders, psutil, numpy as np
from typing import List, Tuple, Union
from pathlib import Path
from PIL import Image
from rich import print
from src.modules.utils import tuple_handler, workers_handler
from tqdm.contrib.concurrent import process_map





class ImageProcessing:
    # Load image
    @staticmethod
    def load(path: str):
        # Check directory if it existed
        if not os.path.exists(path):
            raise FileExistsError("File is not found!!!")
        
        # Load image
        image = cv2.imread(path)

        # Check image
        if image is None:
            raise RuntimeError("Could not load the image file!!!")

        return image
    


    # Resize image to the specified dimensions
    @staticmethod
    def resize(
        image: np.ndarray,
        size: Union[int, List[int], Tuple[int]]
    ):
        
        return cv2.resize(image, tuple_handler(size, 2))
    

    # Adds a border around a given video frame to make it a square frame
    @staticmethod
    def add_border(
        image: np.ndarray, border_color: Tuple | int = (0, 0, 0)
    ):
       
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
        return [cv2.resize(frame, tuple_handler(size, 2)) for frame in video]
    



class DataPreprocessing:
    def __init__(
        self,
        dataset_dir: str,
        save_dir: str,
        ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        image_size: Tuple[int, int] | int = (224, 224),
        sampling_value: int = 0,
        max_frame: int = 0,
        min_frame: int = 0,
        num_workers: int = 0,
    ):
        
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.ratio = ratio
        self.image_size = image_size
        self.sampling_value = sampling_value
        self.max_frame = max_frame
        self.min_frame = min_frame
        self.num_workers = workers_handler(num_workers)
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        self.video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".mpg"]
        


    def __call__(self, save_folder: str= None):
        if not save_folder is None:
            if not os.path.exists(os.path.join(self.save_dir, f"{save_folder}/train")):
                return self.auto(save_folder)
        else:
            if not os.path.exists(os.path.join(self.save_dir, "dataset/train")):
                return self.auto(save_folder)
        
        print("\n[bold][red]The Dataset is processed!!!")
        
    

    # Classes of dataset
    @property
    def classes(self):
        return sorted(os.listdir(self.dataset_dir))
    

    # Process image
    def process_image(self, path: str) -> np.ndarray:
        image = ImageProcessing.load(path)

        file_name = path.split('/')[-1]

        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size:
            result = ImageProcessing.resize(image, self.image_size)

        result = ImageProcessing.add_border(image)

        os.remove(file_name)

        cv2.imwrite(file_name, result)
    

    # Split dataset into train, validation and test dataset
    def split_data(self, input_folder: str, output_folder: str):
        # Check directory if it existed
        if not os.path.exists(input_folder):
            raise FileExistsError("File is not found.")
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        

        # Find the number of validation and test files in dataset
        fixed = tuple(
            [
                int(
                    min(
                        [os.listdir(Path(self.dataset_dir).joinpath(cls)).__len__() for cls in self.classes]
                    ) * r
                )
                for r in self.ratio[1:]
            ]
        )
        

        try:
            splitfolders.fixed(
                input= input_folder,
                output= output_folder,
                seed= 1337,
                fixed= fixed,
                oversample= False,
                group_prefix= None,
                move= False
            )
        except:
            raise RuntimeError("Can not split the dataset!!!")
        
        return True


    # Process video
    def process_video(self, path: str) -> List[np.ndarray]:
        video = VideoProcessing.load(path)

        if self.sampling_value != 0:
            video = VideoProcessing.sampling(np.array(video), self.sampling_value)
        
        if self.max_frame != 0:
            video = VideoProcessing.truncating(np.array(video), self.max_frame)
        
        if self.min_frame != 0:
            video = VideoProcessing.padding(np.array(video), self.min_frame)
        
        if self.image_size:
            video = VideoProcessing.resize(np.array(video), self.image_size)
        
        return video
    

    # Generate frames from video
    def generate_frame(self, path: str):
        # Check directory if it existed
        if not os.path.exists(path):
            raise FileExistsError("File is not found.")
        
        # Get file name
        file_name = Path(path).stem

        # Create destination path
        dst_path = Path(self.save_dir)

        # Process dataset
        video = self.process_video(path)

        for i, frame in enumerate(video):
            save_path = os.path.join(dst_path, f"{file_name}_{i}.jpg")

            if not os.path.exists(save_path):
                image = Image.fromarray(frame)
                image.save(save_path)


    
    # Check file is video or image
    def check_format(self):
        return "video" if Path(os.listdir(self.dataset_dir)[0]).suffix in self.video_extensions else "image"


    def auto(self, save_folder: str= None):
        # Create folder for images of dataset
        if save_folder is None:
            self.save_dir = str(Path(self.save_dir).joinpath("dataset"))
            os.makedirs(self.save_dir, exist_ok= True)
        else:
            self.save_dir = str(Path(self.save_dir).joinpath(save_folder))
            os.makedirs(self.save_dir, exist_ok= True)


        # Calcute chunksize base on cpu parallel power
        benchmark = lambda x: max(
            1, round(len(x) / (self.num_workers * psutil.cpu_freq().max / 1000) / 4)
        )


        print("\n[bold][yellow]Generating data...[/][/]")


        if self.check_format() == "image":
            image_paths = [
                str(image)
                for cls in self.classes
                for ext in self.image_extensions
                for image in Path(self.dataset_dir).joinpath(cls).rglob("*" + ext)
            ]


            # Process image
            process_map(
                self.process_image,
                image_paths,
                max_workers= self.num_workers,
                chunksize= benchmark(image_paths)
            )

            print("\n[bold][green]Processing data successfully!!!")

            # Split dataset
            if self.split_data(self.dataset_dir, self.save_dir):
                print("\n[bold][green]Splitting dataset successfully!!!")
        else:
            video_paths = [
                str(video)
                for ext in self.video_extensions
                for video in Path(self.dataset_dir).rglob("*" + ext)
            ]

            
            # Generate data
            process_map(
                self.generate_frame,
                video_paths,
                max_workers= self.num_workers,
                chunksize= benchmark(video_paths)
            )

        print("\n[bold][green]Complete preprocessing!!![/][/]")