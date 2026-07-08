import logging
import os
import librosa

import numpy as np

logger = logging.getLogger(__name__)

def __min_max_normalize(array):
    """Normalizes a 2D array to values between 0.0 and 1.0 for the CNN."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return array
    return (array - min_val) / (max_val - min_val)


def save_spectrograms(
    audio_path: str,
    chunk_duration: int,
    output_root: str = "visual_embeddings",
    category: str = "",
) -> tuple[str, str]:
    """
    Loads an audio file, cuts it into chunks (e.g., 3 seconds),
    and generates Mel & CQT 2D arrays optimized for CNNs.

    Files are saved as:
        {output_root}/mel/{category}/{songname}_chunkNNN_mel.npy
        {output_root}/cqt/{category}/{songname}_chunkNNN_cqt.npy
    """
    mel_dir = os.path.join(output_root, "mel", category)
    cqt_dir = os.path.join(output_root, "cqt", category)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(cqt_dir, exist_ok=True)

    logger.info(f"Processing: {audio_path}")

    # Standard sample rate for MIR is 22050Hz (good balance of quality and performance)
    sr = 22050

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Calculate how many samples make up our chunk
    samples_per_chunk = int(chunk_duration * sr)
    hop_samples = samples_per_chunk // 2  # 50% overlap sliding window
    total_chunks = max(0, (len(y) - samples_per_chunk) // hop_samples + 1)

    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    vis_mel_path, vis_cqt_path = "", ""

    for i in range(total_chunks):
        # skip end and beginning of full YouTube songs
        if "YouTube" in audio_path and (i == 0 or i == total_chunks - 1):
            continue

        # 1. Slice the audio into a chunk with 50% hop
        start_sample = i * hop_samples
        end_sample = start_sample + samples_per_chunk
        y_chunk = y[start_sample:end_sample]

        # 2. GENERATE MEL-SPECTROGRAM
        # hop_length=512 means the "width" of our image will be roughly 130 pixels per 3 sec
        # n_mels=128 means the "height" of our image will be 128 pixels
        mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to log scale (Decibels)
        mel_normalized = __min_max_normalize(mel_db)  # Scale 0 to 1

        # 3. GENERATE CQT (Constant-Q Transform)
        # n_bins=84 covers exactly 7 octaves of music (12 notes * 7 octaves)
        cqt = librosa.cqt(y=y_chunk, sr=sr, hop_length=512, n_bins=84)
        cqt_mag = np.abs(cqt)  # Get magnitude (discard complex phase)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)  # Convert to Decibels
        cqt_normalized = __min_max_normalize(cqt_db)  # Scale 0 to 1

        # 4. SAVE AS NUMPY ARRAYS (.npy)
        # The CNN will load these much faster than reading PNG files.
        mel_path = os.path.join(mel_dir, f"{song_name}_chunk{i:03d}_mel.npy")
        cqt_path = os.path.join(cqt_dir, f"{song_name}_chunk{i:03d}_cqt.npy")

        if i > total_chunks / 2 and not vis_mel_path:
            vis_mel_path = mel_path
        if i > total_chunks / 2 and not vis_cqt_path:
            vis_cqt_path = cqt_path

        try:
            np.save(mel_path, mel_normalized)
            np.save(cqt_path, cqt_normalized)
        except Exception as e:
            logger.error(f"Failed to save spectrograms for chunk {i}: {e}")
            return "", ""

    logger.info(f"Generated {total_chunks} chunks of Mel and CQT features in '{output_root}'.")

    # Print the shape of the last chunk so you know your CNN input dimensions
    if total_chunks > 0:
        logger.info(f"CNN Input Shape (Mel): {mel_normalized.shape} -> {mel_path}")
        logger.info(f"CNN Input Shape (CQT): {cqt_normalized.shape} -> {cqt_path}")
    else:
        logger.warning(f"Audio too short for a full {chunk_duration}s chunk: {audio_path}")

    return vis_mel_path, vis_cqt_path