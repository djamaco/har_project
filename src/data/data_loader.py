import os
import random
import json
import cv2
from src.config import DATASETS_DIR, PREPROCESSED_IMAGES_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, FRAMES_COUNT, TAKE_NTH_FIRST_CLASSES, TAKE_SPECIFIC_CLASSES, BLACK_WHITE_ONLY

def load_prepared_videos_list_and_mapper():
    with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_NAME}_metadata.json'), 'r') as metadata_file:
        metadata = json.load(metadata_file)
    if TAKE_NTH_FIRST_CLASSES is not None:
        category_mapper = {k: v for k, v in metadata['classes'].items() if k in list(metadata['classes'].keys())[:TAKE_NTH_FIRST_CLASSES]}
    else:
        category_mapper = metadata['classes']
    if TAKE_SPECIFIC_CLASSES is not None:
        category_mapper = {k: v for k, v in category_mapper.items() if v['name'] in TAKE_SPECIFIC_CLASSES}
    
    # filter out the videos that are not in the category mapper
    filtered_videos = {video: label for video, label in metadata['videos'].items() if str(label) in list(category_mapper.keys())}
    videos_list = list(filtered_videos.items())
    random.shuffle(videos_list)

    if (TAKE_NTH_FIRST_CLASSES is None) and (TAKE_SPECIFIC_CLASSES is None): return videos_list, category_mapper

    new_cateogory_mapper = {}
    #Normalize classes labels inside category_mapper and videos_list to be in range [0, len(category_mapper))
    for i, (key, value) in enumerate(category_mapper.items()):
        value['old_label'] = int(key)
        new_cateogory_mapper[str(i)] = value
        for j, (video_name, label) in enumerate(videos_list):
            if label == int(key):
                videos_list[j] = (video_name, i)
    
    return videos_list, new_cateogory_mapper

def extract_preprocessed_frames(videofile_path):
    # Initialize a list to hold the preprocessed frames
    frames_list = []
    # Get a list of preprocessed frames from the given directory
    preprocessed_frames = os.listdir(videofile_path)

    # Iterate over each frame in the directory
    for frame_name in preprocessed_frames:
        # Read the image file to get the frame
        frame = cv2.imread(os.path.join(videofile_path, frame_name))
        # Check if only black and white frames are required; if so, convert to grayscale
        if BLACK_WHITE_ONLY: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame to the desired dimensions (IMAGE_HEIGHT x IMAGE_WIDTH)
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the frame by dividing by 255 to get pixel values in the range [0, 1]
        frame = frame / 255 
        # Append the processed frame to the list
        frames_list.append(frame)
    
    # Return the list of processed frames
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