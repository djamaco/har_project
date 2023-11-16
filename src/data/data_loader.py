import os
import random
import json
import cv2
from src.config import DATASETS_DIR, PREPROCESSED_IMAGES_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, FRAMES_COUNT

def load_prepared_videos_list_and_mapper():
    with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_NAME}_metadata.json'), 'r') as metadata_file:
        metadata = json.load(metadata_file)
    videos_list = list(metadata['videos'].items())
    random.shuffle(videos_list)
    category_mapper = metadata['classes']
    return videos_list, category_mapper

def extract_preprocessed_frames(videofile_path):
    frames_list = []
    preprocessed_frames = os.listdir(videofile_path)
    for frame_name in preprocessed_frames:
        frame = cv2.imread(os.path.join(videofile_path, frame_name))
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255 # Normalize the frame
        frames_list.append(frame)
    return frames_list

def extract_frames_from_videos(videofile_path):
    frames_list = []
    video = cv2.VideoCapture(videofile_path)
    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_number = max(int(video_frames_count / FRAMES_COUNT), 1)
    for frame_number in range(FRAMES_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number * skip_frames_number)
        ok, frame = video.read()
        if not ok: break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255 # Normalize the frame
        frames_list.append(frame)
    video.release()
    return frames_list