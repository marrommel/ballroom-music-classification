import csv
from pathlib import Path
import yt_dlp
from yt_dlp.utils import ExtractorError, DownloadError


def download_song(youtube_url, dance_type):
    """Downloads a YouTube video as a WAV audio file using yt-dlp."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / dance_type
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # yt-dlp options to extract best audio and convert to WAV
    ydl_opts = {
        'retries': 10,  # retry on download errors
        'socket_timeout': 60,  # increase timeout from 20s
        'fragment_retries': 10,
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / 'YouTube-%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    print(f"Downloading from: {youtube_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Return the path to the downloaded WAV file
            return str(output_dir / f"{info['id']}.wav")
    except (ExtractorError, DownloadError):
        print("SKIPPING - Error during extraction occurred")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    songs_file = project_root / "data" / "youtube_songs.txt"

    with open(songs_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dance_type, song_url = row[0], row[1]
            downloaded_file = download_song(song_url, dance_type)
            print(f"\rSaved to: {downloaded_file}\n")
