import os
import tensorflow as tf

from config import TEST_VIDEOS_DIRECTORY, DEFAULT_TRAINED_MODEL_NAME, DEFAULT_YT_VIDEO_URL, MODELS_DIR
from utils.youtube import download_youtube_video
from utils.arguments import get_args
from har.prediction import predict_on_video, predict_single_action

load_model = tf.keras.models.load_model

os.makedirs(TEST_VIDEOS_DIRECTORY, exist_ok=True)
video_title = download_youtube_video(DEFAULT_YT_VIDEO_URL, TEST_VIDEOS_DIRECTORY)

videofile_path = os.path.join(TEST_VIDEOS_DIRECTORY, f'{video_title}.mp4')

args, _ = get_args()

learned_model_name = args.model or DEFAULT_TRAINED_MODEL_NAME
yt_video_url = args.yturl or DEFAULT_YT_VIDEO_URL
output_videofile_path = os.path.join(TEST_VIDEOS_DIRECTORY, f'{video_title}_{learned_model_name}.mp4')
model_path = os.path.join(MODELS_DIR, learned_model_name, 'model.keras')

print(f'Running the script with the following arguments: model={learned_model_name}, yturl={yt_video_url}')
trained_model = load_model(model_path)
predict_on_video(trained_model, videofile_path, output_videofile_path)
print(f'Predictions for the video {videofile_path} were saved to {output_videofile_path}')

predicted_class_name, predicted_label_probability = predict_single_action(trained_model, videofile_path)
print(f'Action prediction for the video {videofile_path}: {predicted_class_name} with probability {predicted_label_probability}')
