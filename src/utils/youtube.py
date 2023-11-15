import os
import sys
from typing import Tuple
from pytube import YouTube

def _progress_function(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining 
    percentage_of_completion = bytes_downloaded / total_size * 100
    sys.stdout.write(f"\rDownloading: {int(percentage_of_completion)}%")
    sys.stdout.flush()

def download_youtube_video(youtube_video_url: str, output_directory: str, redownload: bool = False) -> Tuple[str, str]:
    """
    Downloads a YouTube video from the given URL to the specified output directory.

    Args:
        youtube_video_url (str): The URL of the YouTube video to download.
        output_directory (str): The directory to save the downloaded video to.
        redownload (bool, optional): Whether to redownload the video if it already exists in the output directory. Defaults to False.

    Returns:
        Tuple[str, str]: A tuple containing the title of the downloaded video and the file path of the downloaded video.
    """
    yt = YouTube(youtube_video_url, on_progress_callback=_progress_function)
    title = yt.title
    output_file_path = os.path.join(output_directory, f'{title}.mp4')
    if os.path.exists(output_file_path) and not redownload:
        print(f'Video already downloaded at {output_directory}')
        return title, output_file_path
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video.download(output_directory)
    return title, output_file_path

