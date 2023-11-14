import numpy as np
import tensorflow as tf

from constants.config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, SEED_CONSTANT
from data.data_loader import extract_frames

Sequence = tf.keras.utils.Sequence
to_categorical = tf.keras.utils.to_categorical

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