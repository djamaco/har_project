import os

import os
import pafy

def download_youtube_video(youtube_video_url: str, output_directory: str, redownload: bool = False) -> str:
    """
    Downloads a YouTube video from the given URL and saves it to the specified output directory.

    Args:
        youtube_video_url (str): The URL of the YouTube video to download.
        output_directory (str): The directory to save the downloaded video to.
        redownload (bool, optional): Whether to redownload the video if it already exists in the output directory. Defaults to False.

    Returns:
        str: The title of the downloaded video.
    """
    if os.path.exists(output_directory) and not redownload:
        print(f'Video already downloaded at {output_directory}')
        return output_directory.split('/')[-1].split('.')[0]
    video = pafy.new(youtube_video_url)
    title = video.title
    video_best = video.getbest()
    output_file_path = f'{output_directory}/{title}.mp4'
    video_best.download(filepath=output_file_path, quiet=True)
    return title

