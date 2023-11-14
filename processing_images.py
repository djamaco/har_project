import os
import sys
import random
import json
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Force TensorFlow to use the CPU
tf.config.set_visible_devices([], 'GPU')

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
to_categorical = tf.keras.utils.to_categorical
plot_model = tf.keras.utils.plot_model
mixed_precision = tf.keras.mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# One seed to rule them all
SEED_CONSTANT = 28
np.random.seed(SEED_CONSTANT)
random.seed(SEED_CONSTANT)
tf.random.set_seed(SEED_CONSTANT)

MODELS_DIR = 'models'
DATASETS_DIR = 'datasets'
PREPROCESSED_IMAGES_DIR = 'preprocessed_images'
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
FRAMES_COUNT = 20
BATCH_SIZE = 48
EPOCHS_COUNT = 20

def load_prepared_videos_list_and_mapper():
    with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_DIR}_metadata.json'), 'r') as metadata_file:
        metadata = json.load(metadata_file)
    videos_list = list(metadata['videos'].items())
    random.shuffle(videos_list)
    category_mapper = metadata['classes']
    return videos_list, category_mapper

def extract_frames(videofile_path):
    frames_list = []
    preprocessed_frames = os.listdir(videofile_path)
    for frame_name in preprocessed_frames:
        frame = cv2.imread(os.path.join(videofile_path, frame_name))
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frame = frame / 255 # Normalize the frame
        frames_list.append(frame)
    return frames_list

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, videos_metadata, batch_size, classes_count, shuffle=True):
        self.videos_metadata = videos_metadata
        self.batch_size = batch_size
        self.classes_count = classes_count
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.videos_metadata))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.videos_metadata) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_videos_metadata = [self.videos_metadata[k] for k in batch_indexes]

        X, y = self.__data_generation(batch_videos_metadata)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_videos_metadata):
        X = np.empty((self.batch_size, FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, (videofile_path, category_label) in enumerate(batch_videos_metadata):
            frames = extract_frames(videofile_path)
            if frames is not None:
                X[i,] = np.stack(frames, axis=0)

            y[i] = category_label

        return X, to_categorical(y, num_classes=self.classes_count)

def create_convlstm_model(classes_count):
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (FRAMES_COUNT,
                                                                                      IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))    
    
    model.add(layers.Flatten())
    model.add(layers.Dense(classes_count, activation = "softmax"))

    return model

def plot_metric(model_name, model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    
    plt.savefig(os.path.join('models', model_name, f'{plot_name}.png'))
    plt.clf()
    plt.cla()
    plt.close()

def main():
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
    plot_metric(model_name, convlstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    # Visualize the training and validation accuracy metrices.
    plot_metric(model_name, convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

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