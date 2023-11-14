import os
import json

DATASETS_DIR = 'datasets'
DATASET_NAME = 'UCF-101'

category_names = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME))[:5]
metadata = {'classes': {}, 'videos': {}}

for category_index, category_name in enumerate(category_names):
    videofiles_list = os.listdir(os.path.join(DATASETS_DIR, DATASET_NAME, category_name))
    metadata['classes'][category_index] = {'category_name': category_name, 'total_videos_count': len(videofiles_list)}
    for videofile_name in videofiles_list:
        videofile_path = os.path.join(DATASETS_DIR, DATASET_NAME, category_name, videofile_name)
        metadata['videos'][videofile_path] = category_index
        
with open(os.path.join(DATASETS_DIR, f'{DATASET_NAME}_metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=2)