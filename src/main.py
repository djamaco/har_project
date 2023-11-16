import os
import sys
import random
import datetime as dt
import numpy as np
import shutil
import time

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
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
AdamOptimizer = tf.keras.optimizers.legacy.Adam
CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy

args, _ = get_args()

dl_model_name = args.model or DEFAULT_MODEL_NAME

def main(model_name):
    """
    This function is the main entry point of the program. It creates a model for human activity recognition (HAR)
    using the provided configuration and saves the model to disk. It also generates plots of the model's training
    history and evaluates the model's performance on a validation dataset. If the user specifies the --zip flag,
    the function will also create a zip archive of the saved model.

    Args:
        None

    Returns:
        None
    """
    print('Starting the model creation')
    print(f'Using model {dl_model_name.value if type(dl_model_name) == ModelName else dl_model_name} with provided configuration')
    print('='.join(['' for _ in range(100)]))
    print(f'Random seed={SEED_CONSTANT}')
    print(f'Epochs count={EPOCHS_COUNT}')
    print(f'Batch size={BATCH_SIZE}')
    print(f'Dataset name={DATASET_NAME}')
    print(f'Iamge height={IMAGE_HEIGHT}')
    print(f'Iamge width={IMAGE_WIDTH}')
    print(f'Frames count={FRAMES_COUNT}')
    print(f'Augmenation used={AUGMENTATION_ENABLED}')
    print(f'Model workers count={MODEL_WORKERS_COUNT}')
    print(f'Model max queue size={MODEL_MAX_QUEUE_SIZE}')

    # Load the list of prepared videos and the category mapper
    videos_list, category_mapper = load_prepared_videos_list_and_mapper()
    print(f'Total videos count: {len(videos_list)}')
    print(f'Classes for ptraining: {[i["name"] for i in category_mapper.values()]}')
    print('='.join(['' for _ in range(100)]))

    # Get the number of classes
    classes_count = len(category_mapper)

    # Split the videos into training and testing sets
    train_videos, test_videos = train_test_split(videos_list, test_size=0.3, random_state=SEED_CONSTANT, shuffle=True)
    print(f'Split is done for the videos: {len(train_videos)} videos for training and {len(test_videos)} videos for testing')

    # Create data generators for the training and validation sets
    training_generator = VideoDataGenerator(**{
        'videos_metadata': train_videos,
        'batch_size': BATCH_SIZE,
        'classes_count': classes_count,
        'shuffle': True,
        'augmentation_used': AUGMENTATION_ENABLED,
    })
    validation_generator = VideoDataGenerator(**{
        'videos_metadata': test_videos,
        'batch_size': BATCH_SIZE,
        'classes_count': classes_count,
        'shuffle': False,
        'augmentation_used': AUGMENTATION_ENABLED,
    })

    # Create the model
    model = factory.get_model(dl_model_name, **{'classes_count': classes_count})
    print("Model Created Successfully!")

    # Plot the model's structure and save it to disk
    plot_model(model, to_file = os.path.join(MODELS_DIR, model_name, 'model_structure_plot.png'), show_shapes = True, show_layer_names = True)

    optimizer = AdamOptimizer()
    loss = CategoricalCrossentropy()
    # Compile the model
    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = ["accuracy"],
        experimental_run_tf_function=False,
    )

    # Set up early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # TODO: fix incompleted output log when this callback is triggered
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    
    # Train the model
    start_time = time.time()
    model_training_history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=EPOCHS_COUNT,
        use_multiprocessing=False,
        workers=MODEL_WORKERS_COUNT,
        max_queue_size=MODEL_MAX_QUEUE_SIZE,
        callbacks=[early_stopping, lr_reduction],
    )
    end_time = time.time()

    # Evaluate the model on the validation set
    model_evaluation_history = model.evaluate(validation_generator)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    print(f'Model evaluation loss = {round(model_evaluation_loss, 3)}')
    print(f'Model evaluation accuracy = {round(model_evaluation_accuracy, 3)}')

    # Save the model to disk
    model.save(os.path.join(MODELS_DIR, model_name, 'model.h5'))
    model.save(os.path.join(MODELS_DIR, model_name, 'model.keras'))

    # Generate plots of the model's training history and save them to disk
    create_plot_metric_and_save_to_model(model_name, model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    create_plot_metric_and_save_to_model(model_name, model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    # Print the time taken for training the model
    print(f'Time taken for training the model: {round((end_time - start_time) * 1000, 3)} ms')

    # Rename the model's directory to include the model's accuracy
    accuracy_based_model_name = f'{model_name}_{str(round(model_evaluation_accuracy, 3)).split(".")[1]}'
    os.rename(os.path.join(MODELS_DIR, model_name), os.path.join(MODELS_DIR, accuracy_based_model_name))
    model_name = accuracy_based_model_name

    # If the user specified the --zip flag, create a zip archive of the saved model
    if args.zip and args.zip.lower() == 'true':
        print('Zipping the model')
        shutil.make_archive(os.path.join(MODELS_DIR, model_name), 'zip', os.path.join(MODELS_DIR, model_name))
        print('Zipping is done')

original_stdout = sys.stdout
original_stderr = sys.stderr
try:
    date_time_format = '%Y%m%d%H%M%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_name = f'{dl_model_name.value if type(dl_model_name) == ModelName else dl_model_name}_model_{current_date_time_string}_{EPOCHS_COUNT}'
    os.makedirs(os.path.join(MODELS_DIR, model_name), exist_ok=True)
    sys.stdout = Logger(os.path.join(MODELS_DIR, model_name, 'log.txt'))
    sys.stderr = sys.stdout
    main(model_name)
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr