import numpy as np
import concurrent.futures

from cachetools import cached, LRUCache
from functools import lru_cache

import tensorflow as tf

from src.config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, SEED_CONSTANT, BLACK_WHITE_ONLY
from .data_loader import extract_preprocessed_frames

Sequence = tf.keras.utils.Sequence
to_categorical = tf.keras.utils.to_categorical
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, videos_metadata, batch_size, classes_count, shuffle=True, threads_for_processing_enabled=False, cache_used=False, augmentation_used=True):
        self.videos_metadata = videos_metadata
        self.batch_size = batch_size
        self.classes_count = classes_count
        self.shuffle = shuffle
        self.threads_for_processing_enabled = threads_for_processing_enabled
        self.indexes = np.arange(len(self.videos_metadata))
        self.cache_used = cache_used
        if cache_used: self.cache = LRUCache(maxsize=4096)
        if self.shuffle: np.random.shuffle(self.indexes)
        self.augmentation_used = augmentation_used
        self.augmentation = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

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

    def _process_video(self, videofile_path, category_label, X, y, i):
        if not self.cache_used:
            frames = extract_preprocessed_frames(videofile_path)
        elif videofile_path not in self.cache:
            frames = extract_preprocessed_frames(videofile_path)
            self.cache[videofile_path] = frames
        else:
            frames = self.cache[videofile_path]
    
        if frames is not None:
            if self.augmentation_used:
                frames = [self.augmentation.random_transform(frame) for frame in frames]
            X[i,] = np.stack(frames, axis=0)

        y[i] = category_label

    def __data_generation(self, batch_videos_metadata):
        X = np.empty((self.batch_size, FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32) \
            if BLACK_WHITE_ONLY \
            else np.empty((self.batch_size, FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self.threads_for_processing_enabled:
                futures = []
                for i, (videofile_path, category_label) in enumerate(batch_videos_metadata):
                    futures.append(executor.submit(self._process_video, videofile_path, category_label, X, y, i))
                concurrent.futures.wait(futures)
            else:
                for i, (videofile_path, category_label) in enumerate(batch_videos_metadata):
                    self._process_video(videofile_path, category_label, X, y, i)

        return X, to_categorical(y, num_classes=self.classes_count)