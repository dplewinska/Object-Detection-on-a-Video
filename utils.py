import gdown
from tqdm import tqdm
import json
import os
import cv2
import zipfile


def download_videos():
    ''' Downloads videos'''
    # https://drive.google.com/file/d/1HbyG2hyHeWbrgGInFjy7MSLsJ52YjxEQ/view?usp=sharing
    id = "1HbyG2hyHeWbrgGInFjy7MSLsJ52YjxEQ"
    output = gdown.download(
        f"https://docs.google.com/uc?export=download&id={id}", quiet=False)
    zip_file = f'{output}.zip'

    zip_file = [f for f in os.listdir(
        "./") if f.endswith(('.zip'))][0]
    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('data')

    # Clean up - remove the zip file
    os.remove(zip_file)


def load_settings(file_path):
    with open(file_path, 'r') as file:
        settings = json.load(file)
    return settings


def read_videos(download=True):
    if download:
        download_videos()

    levels = ['easy', 'medium', 'hard']
    video_files = []
    for level in levels:
        video_files += [f"./data/CVP2-data/{level}/"+f for f in os.listdir(
            f"./data/CVP2-data/{level}") if f.endswith(('.mp4', '.mov', '.MOV'))]

    videos = []
    for video_file in video_files:
        video_capture = cv2.VideoCapture(video_file)
        if not video_capture.isOpened():
            print(f"Error opening video file: {video_file}")
            continue
        videos.append(video_capture)

        # ---------------------- reading FPS -------------------------------------------
        # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        # if int(major_ver) < 3:
        #     fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        #     print(
        #         "[video: {}] FPS using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(video_file, fps))
        # else:
        #     fps = video_capture.get(cv2.CAP_PROP_FPS)
        #     print(
        #         "[video: {}] FPS using video.get(cv2.CAP_PROP_FPS) : {}".format(video_file, fps))
        #     num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(f"Total number of frames: {num_frames}")

    return videos


def get_reference_board():
    with open('./data/template.json', 'r') as file:
        data = file.read()
    class_names = {"0": "bazar", "1": "clothes", "2": "household",
                   "3": "newsstand", "4": "food", "5": "delivery", "6": "furniture", "7": "cards"}
    annotations = json.loads(data)["annotations"]
    bboxes = [list(map(int, annotation["bbox"])) for annotation in annotations]
    bboxes_names = [class_names[str(annotation["category_id"])]
                    for annotation in annotations]
    image_path = "./data/" + json.loads(data)["images"][0]["file_name"]
    image = cv2.imread(image_path)
    return image, bboxes, bboxes_names
