import logging
import os
import sys

from helpers.spectrograms import save_spectrograms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# configuration
DATA_DIR = "train_data_set"
OUTPUT_ROOT = "image_embeddings"
CHUNK_DURATION = 15
AUDIO_EXTENSIONS = (".mp3", ".wav")


def build_dataset() -> None:
    """
    Walk the data/ directory and generate Mel & CQT spectrograms for every audio
    file found.

    Output path: image_embeddings/[mel|cqt]/[dance_type]/[songname_chunkNNN_melORcqt.npy]
    """
    if not os.path.isdir(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found. Nothing to process.")
        return

    processed, failed = 0, 0

    for root, _dirs, files in os.walk(DATA_DIR):
        audio_files = [f for f in files if f.lower().endswith(AUDIO_EXTENSIONS)]
        if not audio_files:
            continue

        # The dance type is the folder name directly under data/
        rel = os.path.relpath(root, DATA_DIR)
        dance_type = rel.split(os.sep)[0] if rel != "." else ""

        if not dance_type:
            logger.warning(f"Skipping audio files directly in '{DATA_DIR}' (no dance label).")
            continue

        for file_name in audio_files:
            audio_path = os.path.join(root, file_name)
            try:
                mel_path, cqt_path, temp_path = save_spectrograms(
                    audio_path,
                    CHUNK_DURATION,
                    output_root=OUTPUT_ROOT,
                    category=dance_type,
                )
                if not (mel_path and cqt_path and temp_path):
                    logger.error(f"No spectrograms produced for {audio_path}")
                    failed += 1
                    continue
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                failed += 1

    logger.info(f"Done. Processed {processed} file(s), {failed} failure(s).")


if __name__ == "__main__":
    build_dataset()
