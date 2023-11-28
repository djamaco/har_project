from collections import deque
import numpy as np
from typing import Any, Tuple

import cv2
import tensorflow as tf

from src.config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH
from src.data.data_loader import load_prepared_videos_list_and_mapper
from src.utils.helper import sanitize_file_path

Sequential = tf.keras.models.Sequential

def predict_on_video(model: Sequential, video_file_path: str, output_file_path: str) -> None:
    """
    Predicts the activity class of a human performing an action in a video file and writes the output to a new video file.
    
    Args:
    - model: A trained Keras Sequential model.
    - video_file_path: A string representing the path to the input video file.
    - output_file_path: A string representing the path to the output video file.
    
    Returns:
    - None
    """
 
    # Open the input video file
    video_reader = cv2.VideoCapture(sanitize_file_path(video_file_path))
 
    # Get the original video dimensions
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Create a new video file for the output
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
 
    # Create a deque to hold the last FRAMES_COUNT frames
    frames_queue = deque(maxlen = FRAMES_COUNT)
    
    # Initialize the predicted class name to an empty string
    predicted_class_name = ''

    # Load the list of activity classes and their corresponding integer labels
    _, classes = load_prepared_videos_list_and_mapper()
 
    # Loop through each frame in the input video file
    while video_reader.isOpened():
        # Read the next frame from the video file
        ok, frame = video_reader.read() 
        if not ok: break
 
        # Resize the frame to the desired dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the pixel values to be between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Add the normalized frame to the deque
        frames_queue.append(normalized_frame)
 
        # If the deque is full, predict the activity class of the last FRAMES_COUNT frames
        if len(frames_queue) == FRAMES_COUNT:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = classes[str(predicted_label)]['name']
 
        # Add the predicted class name as text to the current frame
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        # Write the current frame to the output video file
        video_writer.write(frame)
        
    # Release the input and output video files
    video_reader.release()
    video_writer.release()

def predict_single_action(model: Sequential, video_file_path: str) -> Tuple[str, Any]:
    """
    Predicts the action in a single video file using a trained model.

    Args:
        model (Sequential): A trained Keras model.
        video_file_path (str): The path to the video file.

    Returns:
        A tuple containing the predicted action label and the probability of the prediction.
    """
    video_reader = cv2.VideoCapture(sanitize_file_path(video_file_path))
    frames_list = []
    predicted_class_name = ''
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / FRAMES_COUNT), 1)
 
    for frame_counter in range(FRAMES_COUNT):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        ok, frame = video_reader.read() 
        if not ok: break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
 
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]
    print(predicted_labels_probabilities)
    predicted_label = np.argmax(predicted_labels_probabilities)

    _, classes = load_prepared_videos_list_and_mapper()
    predicted_class_name = classes[str(predicted_label)]['name']
    
    video_reader.release()
    return predicted_class_name, predicted_labels_probabilities[predicted_label]