from prepare_video import get_prepared_params
from src.har.prediction import predict_single_action

trained_model, videofile_path, output_videofile_path, learned_model_name = get_prepared_params()

print(f'Running the classification of the video script with the following arguments: model={learned_model_name}, video path={videofile_path}')
predicted_class_name, predicted_label_probability = predict_single_action(trained_model, videofile_path)
print(f'Action prediction for the video {videofile_path}: {predicted_class_name} with probability {predicted_label_probability}')