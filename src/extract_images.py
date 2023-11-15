import json
import concurrent.futures

from config import *
from data.data_extractor import process_class_videos

classes = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME))
metadata = {'classes': {}, 'videos': {}}

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for class_index, class_name in enumerate(classes):
        videofiles_list = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME, class_name))
        metadata['classes'][class_index] = {'name': class_name, 'total_videos_count': 0}
        futures.append(executor.submit(process_class_videos, class_name, class_index, videofiles_list, classes, metadata))
    concurrent.futures.wait(futures)

with open(os.path.join(DATASETS_DIR, f'{PREPROCESSED_IMAGES_DIR}_metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=2)