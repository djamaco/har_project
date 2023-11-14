"""This module checks that the used dataset is correct and visualizes it."""
import os
import random
import cv2
import matplotlib.pyplot as plt

DATASET_NAME = 'UCF-101'

plt.figure(figsize=(20, 20))

category_names = os.listdir(DATASET_NAME)

def show_categories_info() -> None:
    for category_name in category_names:
        videos_count = len(os.listdir(f'{DATASET_NAME}/{category_name}'))
        print(f'Count of {category_name} videos = {videos_count}')

def generate_category_image(out_file: str, frm: int, to: int) -> None:
    for counter_number, category_name in enumerate(category_names[frm:to], 1):
        category_video_names_list = os.listdir(f'{DATASET_NAME}/{category_name}')
        selected_video_name = random.choice(category_video_names_list)
        video = cv2.VideoCapture(f'{DATASET_NAME}/{category_name}/{selected_video_name}')

        _, first_frame = video.read()
        video.release()

        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(first_frame, category_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 10), 3)

        plt.subplot(5, 4, counter_number)
        plt.imshow(first_frame)
        plt.axis('off')

    plt.savefig(os.path.join('check_images', out_file))

show_categories_info()
generate_category_image('check_1.png', 0, 20)
generate_category_image('check_2.png', 20, 40)
generate_category_image('check_3.png', 40, 60)
generate_category_image('check_4.png', 60, 80)
generate_category_image('check_5.png', 80, 100)
