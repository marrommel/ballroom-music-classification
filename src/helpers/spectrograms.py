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


def compute_spectrograms(audio: np.ndarray, sample_rate: int, spec_types: list[str]) -> dict[str, np.ndarray]:
    """Compute min-max normalized spectrograms for the requested types only.

    Args:
        audio: 1-D float32 audio samples.
        sample_rate: Sample rate of the audio file.
        spec_types: Which spectrograms to compute. Supported: 'mel', 'cqt', 'temp'.

    Returns:
        Dict mapping each requested spec type to its normalized np.ndarray.

    Raises:
        ValueError: If an unknown spec type is requested.
    """
    specs: dict[str, np.ndarray] = {}
    _onset_env = None  # computed lazily, shared if 'temp' is requested

    for spec_type in spec_types:
        if spec_type == "mel":
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            specs["mel"] = __min_max_normalize(mel_db).astype(np.float32)

        elif spec_type == "cqt":
            cqt = librosa.cqt(y=audio, sr=sample_rate, hop_length=512, n_bins=84)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            specs["cqt"] = __min_max_normalize(cqt_db).astype(np.float32)

        elif spec_type == "temp":
            if _onset_env is None:
                _onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=512)
            tempogram = librosa.feature.tempogram(
                onset_envelope=_onset_env, sr=sample_rate, hop_length=512, win_length=192
            )
            specs["temp"] = __min_max_normalize(tempogram).astype(np.float32)

        else:
            raise ValueError(f"Unknown spec type '{spec_type}'. Supported: 'mel', 'cqt', 'temp'.")

    return specs


def save_spectrograms(
        audio_path: str,
        chunk_duration: int,
        output_root: str = "image_embeddings",
        category: str = "",
) -> tuple[str, str, str]:
    """
    Loads an audio file, cuts it into chunks (e.g., 3 seconds),
    and generates Mel & CQT 2D arrays optimized for CNNs.

    Files are saved as:
        {output_root}/mel/{category}/{songname}_chunkNNN_mel.npy
        {output_root}/cqt/{category}/{songname}_chunkNNN_cqt.npy
    """
    mel_dir = os.path.join(output_root, "mel", category)
    cqt_dir = os.path.join(output_root, "cqt", category)
    temp_dir = os.path.join(output_root, "temp", category)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(cqt_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"Processing: {audio_path}")

    # Load the audio file
    sample_rate = 22050
    y, sample_rate = librosa.load(audio_path, sr=sample_rate)

    specs = compute_spectrograms(y, sample_rate, ["mel", "cqt", "temp"])
    mel_spec, cqt_spec, temp_spec = specs["mel"], specs["cqt"], specs["temp"]

    # Calculate chunk sie in spectrogram frames
    frames_per_second = sample_rate / 512
    frames_per_chunk = int(chunk_duration * frames_per_second)
    hop_frames = frames_per_chunk // 2  # 50% overlap in frames

    # Calculate the number of chunks from the time axis of one spectrogram
    total_frames = mel_spec.shape[1]
    total_chunks = max(0, (total_frames - frames_per_chunk) // hop_frames + 1)


    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    vis_mel_path, vis_cqt_path, vis_temp_path = "", "", ""

    for i in range(total_chunks):
        # Slice the audio into a chunk with 50% hop
        start_frame = i * hop_frames
        end_frame = start_frame + frames_per_chunk

        mel_chunk = mel_spec[:, start_frame:end_frame]
        cqt_chunk = cqt_spec[:, start_frame:end_frame]
        temp_chunk = temp_spec[:, start_frame:end_frame]

        # save spectrograms as numpy arrays to avoid loading PNG files
        mel_path = os.path.join(mel_dir, f"{song_name}_chunk{i:03d}_mel.npy")
        cqt_path = os.path.join(cqt_dir, f"{song_name}_chunk{i:03d}_cqt.npy")
        temp_path = os.path.join(temp_dir, f"{song_name}_chunk{i:03d}_temp.npy")

        if i > total_chunks / 2 and not vis_mel_path:
            vis_mel_path = mel_path
        if i > total_chunks / 2 and not vis_cqt_path:
            vis_cqt_path = cqt_path
        if i > total_chunks / 2 and not vis_temp_path:
            vis_temp_path = temp_path

        try:
            np.save(mel_path, mel_chunk)
            np.save(cqt_path, cqt_chunk)
            np.save(temp_path, temp_chunk)
        except Exception as e:
            logger.error(f"Failed to save spectrograms for chunk {i}: {e}")
            return "", ""

    logger.info(f"Generated {total_chunks} chunks of Mel, CQT and Tempo features in '{output_root}'.")

    # Print the shape of the last chunk so you know your CNN input dimensions
    if total_chunks > 0:
        pass
        #logger.info(f"CNN Input Shape (Mel): {mel_normalized.shape} -> {mel_path}")
        #logger.info(f"CNN Input Shape (CQT): {cqt_normalized.shape} -> {cqt_path}")
        #logger.info(f"CNN Input Shape (Tempo): {temp_normalized.shape} -> {temp_path}")
    else:
        logger.warning(f"Audio too short for a full {chunk_duration}s chunk: {audio_path}")

    return vis_mel_path, vis_cqt_path, vis_temp_path