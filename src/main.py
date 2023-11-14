import os
import sys
import random
import datetime as dt
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from constants.config import *
from data.data_loader import load_prepared_videos_list_and_mapper
from data.video_data_generator import VideoDataGenerator
from models.convlstm_model import create_convlstm_model
from utils.logger import Logger
from utils.plot_utils import create_plot_metric_and_save_to_model

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

def main():
    print('Starting the model creation')
    print(f'Random seed=')
    # Use the data generator
    videos_list, category_mapper = load_prepared_videos_list_and_mapper()
    classes_count = len(category_mapper)

    train_videos, test_videos = train_test_split(videos_list, test_size=0.3, random_state=SEED_CONSTANT, shuffle=True)

    print(f'Split is done for the videos: {len(train_videos)} videos for training and {len(test_videos)} videos for testing')

    # Create the generators
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

    # Construct the required convlstm model.
    convlstm_model = create_convlstm_model(classes_count)
    print("Model Created Successfully!")

    # Plot the structure of the contructed model.
    # plot_model(convlstm_model, to_file = 'convlstm_model_structure_plot.png', show_shapes = True, show_layer_names = True)

    convlstm_model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'Adam',
        metrics = ["accuracy"],
        experimental_run_tf_function=False,
    )
    convlstm_model_training_history = convlstm_model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=EPOCHS_COUNT,
        use_multiprocessing=False,
        workers=4,
        max_queue_size=10,
    )
    model_evaluation_history = convlstm_model.evaluate(validation_generator)

    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    print(f'Model evaluation loss = {round(model_evaluation_loss, 3)}')
    print(f'Model evaluation accuracy = {round(model_evaluation_accuracy, 3)}')

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = os.path.join(MODELS_DIR, model_name, 'model.keras')

    # Save your Model.
    convlstm_model.save(model_file_name)

    # Visualize the training and validation loss metrices.
    create_plot_metric_and_save_to_model(model_name, convlstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    # Visualize the training and validation accuracy metrices.
    create_plot_metric_and_save_to_model(model_name, convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    # TODO: possibly add ziping the model's folder

original_stdout = sys.stdout
original_stderr = sys.stderr
try:
    date_time_format = '%Y%m%d%H%M%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_name = f'convlstm_model_{current_date_time_string}_{EPOCHS_COUNT}'
    os.makedirs(os.path.join(MODELS_DIR, model_name), exist_ok=True)
    sys.stdout = Logger(os.path.join(MODELS_DIR, model_name, 'log.txt'))
    sys.stderr = sys.stdout
    main()
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr