import os
from .constants import ModelName

# One seed to rule them all
SEED_CONSTANT = 28 # TODO: add a comment WHY

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_DIR, 'processed_models')
DATASETS_DIR =  os.path.join(PROJECT_DIR, 'datasets')
DATASET_NAME = 'UCF-101'
PREPROCESSED_IMAGES_NAME = 'preprocessed_images'
PREPROCESSED_IMAGES_DIR = os.path.join(DATASETS_DIR, PREPROCESSED_IMAGES_NAME)

DEFAULT_MODEL_NAME = ModelName.CONVLSTM

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
FRAMES_COUNT = 20
BATCH_SIZE = 128
EPOCHS_COUNT = 1