from prepare_video import get_prepared_params
from har.prediction import predict_on_video

trained_model, videofile_path, output_videofile_path, learned_model_name = get_prepared_params()

print(f'Running the full video analysis script with the following arguments: model={learned_model_name}, video path={videofile_path}')
predict_on_video(trained_model, videofile_path, output_videofile_path)
print(f'Predictions for the video {videofile_path} were saved to {output_videofile_path}')