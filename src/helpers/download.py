import os
import yt_dlp


def download_song(youtube_url, output_dir="audio_files"):
    """Downloads a YouTube video as a WAV audio file using yt-dlp."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yt-dlp options to extract best audio and convert to WAV
    ydl_opts = {
        'retries': 10,  # retry on download errors
        'socket_timeout': 60,  # increase timeout from 20s
        'fragment_retries': 10,
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    print(f"Downloading from: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        # Return the path to the downloaded WAV file
        return os.path.join(output_dir, f"{info['id']}.wav")