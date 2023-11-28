import os
import cv2

from src.config import *


def extract_and_save_frames(videofile_path, video_name, class_name, class_index, video_index, total_video_count, classes):
    print(f'Extracting frames for {video_name} of class {class_name} [{class_index + 1}/{len(classes)}][{video_index + 1}/{total_video_count}]')
    # Open the video file using OpenCV
    video = cv2.VideoCapture(videofile_path)
    # Get the total number of frames in the video
    video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the number of frames to skip to evenly sample across the video
    skip_frames_number = max(int(video_frames_count / FRAMES_COUNT), 1)
    # Create the directory to save the preprocessed images, if it doesn't exist
    os.makedirs(os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name), exist_ok=True)
    
    # Iterate through the specified number of frames
    for frame_index in range(FRAMES_COUNT):
        # Set the position of the next frame to read
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index * skip_frames_number)
        # Read the frame from the video
        ok, frame = video.read()
        # If reading the frame failed, break out of the loop
        if not ok: break
        # Resize the frame to the desired dimensions
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Construct the filename for the frame
        frame_filename = os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name, f'frame{frame_index + 1}.png')
        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)
    
    # Release the video file
    video.release()

def process_class_videos(class_name, class_index, videofiles_list, classes, metadata):
    total_video_count = len(videofiles_list)
    metadata['classes'][class_index]['total_videos_count'] = total_video_count
    for video_index, video_name in enumerate(videofiles_list[:FRAMES_COUNT]): # [:FRAMES_COUNT] is added here just to be consistent with all the rest code. before each run of a main.py we need to make sure that images were pre-processed with the same size.
        videofile_path = os.path.join(DATASETS_DIR, DATASET_NAME, class_name, video_name)
        extract_and_save_frames(videofile_path, video_name, class_name, class_index, video_index, total_video_count, classes)
        metadata['videos'][os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_DIR, class_name, video_name)] = class_index
