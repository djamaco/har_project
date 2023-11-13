import os
import json

DATASET_NAME = 'datasets/UCF-101'

category_names = os.listdir(DATASET_NAME)[:5]
metadata = {'classes': {}, 'videos': {}}

for category_index, category_name in enumerate(category_names):
    videofiles_list = os.listdir(os.path.join(DATASET_NAME, category_name))
    metadata['classes'][category_index] = {'category_name': category_name, 'total_videos_count': len(videofiles_list)}
    for videofile_name in videofiles_list:
        videofile_path = os.path.join(DATASET_NAME, category_name, videofile_name)
        metadata['videos'][videofile_path] = category_index
        
with open('metadata.json', 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=2)