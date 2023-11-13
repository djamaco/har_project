import os
import json
import cv2
import concurrent.futures

DATASETS_DIR = 'datasets'
DATASET_NAME = 'UCF-101'
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
FRAMES_COUNT = 20
PREPROCESSED_IMAGES_DIR = 'preprocessed_images'

classes = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME))[:5]
metadata = {'classes': {}, 'videos': {}}

def extract_and_save_frames(videofile_path, video_name, class_name, class_index, video_index, total_video_count):
    print(f'Extracting frames for {video_name} of class {class_name} [{class_index + 1}/{len(classes)}][{video_index + 1}/{total_video_count}]')
    video = cv2.VideoCapture(videofile_path)
    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_number = max(int(video_frames_count / FRAMES_COUNT), 1)
    os.makedirs(os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name), exist_ok=True)
    for frame_index in range(FRAMES_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index * skip_frames_number)
        ok, frame = video.read()
        if not ok: break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Save the frame as an image file
        frame_filename = os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name, f'frame{frame_index + 1}.png')
        cv2.imwrite(frame_filename, frame)
    video.release()

def process_class_videos(class_name, class_index, videofiles_list):
    total_video_count = len(videofiles_list)
    metadata['classes'][class_index]['total_videos_count'] = total_video_count
    for video_index, video_name in enumerate(videofiles_list):
        videofile_path = os.path.join(DATASETS_DIR, DATASET_NAME, class_name, video_name)
        extract_and_save_frames(videofile_path, video_name, class_name, class_index, video_index, total_video_count)
        metadata['videos'][os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name)] = class_index

with concurrent.futures.ThreadPoolExecutor() as executor:
    features_to_wait = []
    for class_index, class_name in enumerate(classes):
        videofiles_list = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME, class_name))
        metadata['classes'][class_index] = {'name': class_name, 'total_videos_count': 0}
        features_to_wait.append(executor.submit(process_class_videos, class_name, class_index, videofiles_list))
    concurrent.futures.wait(features_to_wait)

with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_DIR}_metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=2)