import os
import random
import json
import cv2
from constants.config import DATASETS_DIR, PREPROCESSED_IMAGES_NAME, SEED_CONSTANT, IMAGE_HEIGHT, IMAGE_WIDTH

def load_prepared_videos_list_and_mapper():
    with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_NAME}_metadata.json'), 'r') as metadata_file:
        metadata = json.load(metadata_file)
    videos_list = list(metadata['videos'].items())
    random.shuffle(videos_list)
    category_mapper = metadata['classes']
    return videos_list, category_mapper

def extract_frames(videofile_path):
    frames_list = []
    preprocessed_frames = os.listdir(videofile_path)
    for frame_name in preprocessed_frames:
        frame = cv2.imread(os.path.join(videofile_path, frame_name))
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255 # Normalize the frame
        frames_list.append(frame)
    return frames_list