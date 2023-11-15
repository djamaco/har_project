import os
import sys
import random
import datetime as dt
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from config import *
from data.data_loader import load_prepared_videos_list_and_mapper
from data.video_data_generator import VideoDataGenerator
from models import factory
from utils.logger import Logger
from utils.plot_utils import create_plot_metric_and_save_to_model
from utils.arguments import get_args

# Force TensorFlow to use the CPU
tf.config.set_visible_devices([], 'GPU')

np.random.seed(SEED_CONSTANT)
random.seed(SEED_CONSTANT)
tf.random.set_seed(SEED_CONSTANT)

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
to_categorical = tf.keras.utils.to_categorical
plot_model = tf.keras.utils.plot_model
mixed_precision = tf.keras.mixed_precision

args, _ = get_args()

dl_model_name = args.model or DEFAULT_MODEL_NAME

def main():
    # TODO: add a usefull printing here so all the config and init params will be logged
    print('Starting the model creation')
    print(f'Random seed=')

    videos_list, category_mapper = load_prepared_videos_list_and_mapper()
    classes_count = len(category_mapper)
    train_videos, test_videos = train_test_split(videos_list, test_size=0.3, random_state=SEED_CONSTANT, shuffle=True)
    print(f'Split is done for the videos: {len(train_videos)} videos for training and {len(test_videos)} videos for testing')

    training_generator = VideoDataGenerator(**{
        'videos_metadata': train_videos,
        'batch_size': BATCH_SIZE,
        'classes_count': classes_count,
        'shuffle': True
    })
    validation_generator = VideoDataGenerator(**{
        'videos_metadata': test_videos,
        'batch_size': BATCH_SIZE,
        'classes_count': classes_count,
        'shuffle': False
    })

    model = factory.get_model(dl_model_name, **{'classes_count': classes_count})
    print("Model Created Successfully!")

    # TODO: enable it and save to the model directory
    # Plot the structure of the contructed model.
    # plot_model(convlstm_model, to_file = 'convlstm_model_structure_plot.png', show_shapes = True, show_layer_names = True)

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'Adam',
        metrics = ["accuracy"],
        experimental_run_tf_function=False,
    )
    model_training_history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=EPOCHS_COUNT,
        use_multiprocessing=False,
        workers=4,
        max_queue_size=10,
    )
    model_evaluation_history = model.evaluate(validation_generator)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    print(f'Model evaluation loss = {round(model_evaluation_loss, 3)}')
    print(f'Model evaluation accuracy = {round(model_evaluation_accuracy, 3)}')

    model_file_name = os.path.join(MODELS_DIR, model_name, 'model.keras')
    model.save(model_file_name)

    create_plot_metric_and_save_to_model(model_name, model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    create_plot_metric_and_save_to_model(model_name, model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    # TODO: possibly add ziping the model's folder

original_stdout = sys.stdout
original_stderr = sys.stderr
try:
    date_time_format = '%Y%m%d%H%M%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_name = f'{dl_model_name.value if isinstance(dl_model_name, ModelName) else dl_model_name}_model_{current_date_time_string}_{EPOCHS_COUNT}'
    os.makedirs(os.path.join(MODELS_DIR, model_name), exist_ok=True)
    sys.stdout = Logger(os.path.join(MODELS_DIR, model_name, 'log.txt'))
    sys.stderr = sys.stdout
    main()
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr