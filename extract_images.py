import os
import json
import cv2

DATASETS_DIR = 'datasets'
DATASET_NAME = 'UCF-101'
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
FRAMES_COUNT = 20
PREPROCESSED_IMAGES_DIR = 'preprocessed_images'

classes = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME))[:2]
metadata = {'classes': {}, 'images': {}}

def extract_and_save_frames(videofile_path, name, class_name, class_index, video_index, total_video_count):
    print(f'Extracting frames for {name} of class {class_name} [{class_index + 1}/{len(classes)}][{video_index + 1}/{total_video_count}]')
    video = cv2.VideoCapture(videofile_path)
    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_number = max(int(video_frames_count / FRAMES_COUNT), 1)
    for frame_number in range(FRAMES_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number * skip_frames_number)
        ok, frame = video.read()
        if not ok: break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Save the frame as an image file
        frame_filename = os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, f'{os.path.basename(videofile_path)}_frame{frame_number}.png')
        metadata['images'][frame_filename] = class_index
        metadata['classes'][class_index]['total_images_count'] += 1
        cv2.imwrite(frame_filename, frame)
    video.release()

for class_index, class_name in enumerate(classes):
    print(f'Processing videos of {class_name} class [{class_index + 1}/{len(classes)}]]')
    videofiles_list = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME, class_name))
    metadata['classes'][class_index] = {'name': class_name, 'total_images_count': 0}
    os.makedirs(os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name), exist_ok=True)
    for video_index, videofile_name in enumerate(videofiles_list):
        videofile_path = os.path.join(DATASETS_DIR, DATASET_NAME, class_name, videofile_name)
        extract_and_save_frames(videofile_path, videofile_name, class_name, class_index, video_index, len(videofiles_list))

with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_DIR}_metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=2)