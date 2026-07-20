import csv
import re
from pathlib import Path
import yt_dlp
from yt_dlp.utils import ExtractorError, DownloadError


def sanitize_song_name(name: str) -> str:
    """Converts a song name to a file-system safe slug (e.g. 'Over My Shoulder' -> 'Over-My-Shoulder')."""
    name = name.strip()
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'[^A-Za-z0-9-]+', '', name)
    return name


def download_song(youtube_url, dance_type, song_name):
    """Downloads a YouTube video as a WAV audio file using yt-dlp."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "assets/test" / dance_type
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    safe_name = sanitize_song_name(song_name)

    # yt-dlp options to extract the best audio and convert to WAV
    ydl_opts = {
        'retries': 10,  # retry on download errors
        'socket_timeout': 60,  # increase timeout from 20s
        'fragment_retries': 10,
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / f'{safe_name}-%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    print(f"Downloading '{song_name}' from: {youtube_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Return the path to the downloaded WAV file
            return str(output_dir / f"{safe_name}-{info['id']}.wav")
    except (ExtractorError, DownloadError) as e:
        print("SKIPPING - Error during extraction occurred")
        print(e)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    songs_file = project_root / "assets/train_data_set" / "youtube_songs.txt"

    with open(songs_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dance_type, song_name, song_url = row[0], row[1], row[2]
            downloaded_file = download_song(song_url, dance_type, song_name)
            print(f"\rSaved to: {downloaded_file}\n")
