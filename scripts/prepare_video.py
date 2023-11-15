import os

import tensorflow as tf

from src.config import TEST_VIDEOS_DIRECTORY, DEFAULT_TRAINED_MODEL_NAME, DEFAULT_YT_VIDEO_URL, MODELS_DIR, MODEL_FORMAT
from src.utils.arguments import get_args
from src.utils.youtube import download_youtube_video

load_model = tf.keras.models.load_model

args, _ = get_args()

def get_prepared_params():
    os.makedirs(TEST_VIDEOS_DIRECTORY, exist_ok=True)
    video_title, videofile_path = download_youtube_video(DEFAULT_YT_VIDEO_URL, TEST_VIDEOS_DIRECTORY)
    print(f'Video {video_title} was downloaded to {TEST_VIDEOS_DIRECTORY}')
    print(f'Video file path: {videofile_path}')
    
    learned_model_name = args.model or DEFAULT_TRAINED_MODEL_NAME
    yt_video_url = args.yturl or DEFAULT_YT_VIDEO_URL
    output_videofile_path = os.path.join(TEST_VIDEOS_DIRECTORY, f'{video_title}_{learned_model_name}.mp4')
    model_path = os.path.join(MODELS_DIR, learned_model_name, 'model.{MODEL_FORMAT}')

    
    trained_model = load_model(model_path)

    return trained_model, videofile_path, output_videofile_path, learned_model_name