import argparse
import logging
import math
import sys

import librosa
import numpy as np
import torch

from cnn.model import DualSpectrogramClassificationModel
from helpers.spectrograms import compute_spectrogram_pair

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 'NoDance',
DANCE_CLASSES = ['DiscoFox', 'ChaChaCha', 'Rumba', 'Jive', 'Quickstep', 'Tango', 'VienneseWaltz', 'Waltz']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "best_vision_model.pt"
SAMPLE_RATE = 22050

CHUNK_DURATION = 15 # must be SAME as the one used for training
INFERENCE_DURATION = 30   # seconds of audio to analyse at inference time (independent of CHUNK_DURATION)
MIN_DURATION = 3


def load_model(checkpoint_path: str) -> DualSpectrogramClassificationModel:
    """Load a trained DualStreamVisionNet from a checkpoint file.

    Args:
        checkpoint_path: Path to the trained state-dict file.

    Returns:
        The model in eval mode, ready for inference.
    """
    model = DualSpectrogramClassificationModel(num_classes=len(DANCE_CLASSES)).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def validate_durations(total_duration: int) -> bool:
    if total_duration < MIN_DURATION:
        raise ValueError(f"Audio is too short ({total_duration:.1f}s < {MIN_DURATION}s). Skipping.")

    inference_duration = min(INFERENCE_DURATION, total_duration)
    if inference_duration < MIN_DURATION:
        raise ValueError(
            f"Inference duration ({inference_duration:.1f}s) is below the minimum of {MIN_DURATION}s. Skipping."
        )

    if inference_duration < CHUNK_DURATION:
        logger.warning(
            f"Inference duration ({inference_duration:.1f}s) is shorter than the training chunk size "
            f"({CHUNK_DURATION}s). Prediction quality may be reduced."
        )
        return False

    return True

def extract_chunks(audio_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load audio, center a window of *inference_duration* seconds, and return
    (mel, cqt) pairs for each CHUNK_DURATION-sized slice with 50 % overlap.

    Returns:
        List of (mel, cqt) array pairs — one entry per chunk.

    Raises:
        ValueError: if the audio or requested duration is below MIN_DURATION.
    """
    total_duration = math.floor(librosa.get_duration(path=audio_path))
    if not validate_durations(total_duration):
        return []
    else:
        inference_duration = min(INFERENCE_DURATION, total_duration)

    # Center the inference window inside the track
    centre = total_duration / 2
    offset = max(0.0, min(centre - inference_duration / 2, total_duration - inference_duration))
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, offset=offset, duration=inference_duration)

    samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)

    # for short inference_duration return only one chunk
    if len(y) < samples_per_chunk:
        return [compute_spectrogram_pair(y, SAMPLE_RATE)]

    # Sliding-window chunking for long inference_duration
    hop_samples = samples_per_chunk // 2
    total_chunks = (len(y) - samples_per_chunk) // hop_samples + 1
    return [
        compute_spectrogram_pair(y[i * hop_samples: i * hop_samples + samples_per_chunk], SAMPLE_RATE)
        for i in range(total_chunks)
    ]


def generate_spectrograms(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an audio file and generate mel and CQT spectrograms from a 30-second middle chunk.

    Args:
        audio_path: Path to the input audio file (.wav, .mp3, etc.).

    Returns:
        Tuple of (mel_array, cqt_array), each a 2D float32 numpy array.

    Raises:
        ValueError: If the audio is shorter than ``CHUNK_DURATION`` seconds.
    """
    total_duration = librosa.get_duration(path=audio_path)
    if total_duration < CHUNK_DURATION:
        raise ValueError(
            f"Audio is too short: {total_duration:.1f}s, need at least {CHUNK_DURATION}s."
        )

    offset = min(total_duration / 2, total_duration - CHUNK_DURATION)
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, offset=offset, duration=CHUNK_DURATION)

    return compute_spectrogram_pair(y, SAMPLE_RATE)

def predict(model: DualSpectrogramClassificationModel,  chunks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str, dict[str, float]]:
    """Run inference on multiple (mel, cqt) spectrogram pairs.

    Args:
        model: A loaded DualStreamVisionNet in eval mode.
        chunks: List of mel and cqt spectrogram pairs.

    Returns:
        Tuple of (predicted_class_name, {class_name: probability}).
    """
    all_probs = []
    for mel, cqt in chunks:
        mel_img = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        cqt_img = torch.tensor(cqt).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(mel_img, cqt_img)
            all_probs.append(torch.softmax(logits, dim=1).squeeze(0).cpu())

    avg_probs = torch.stack(all_probs).mean(dim=0).cpu().tolist()
    predicted_idx = int(np.argmax(avg_probs))
    prob_map = {cls: round(p, 4) for cls, p in zip(DANCE_CLASSES, avg_probs)}
    return DANCE_CLASSES[predicted_idx], prob_map


def main() -> None:
    """CLI entry point: classify the dance style of an audio file."""
    parser = argparse.ArgumentParser(description="Ballroom dance style classifier")
    parser.add_argument("audio", help="Path to the audio file to classify (.wav, .mp3, etc.)")
    args = parser.parse_args()

    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)

    print(f"Generating spectrograms for: {args.audio}")
    try:
        chunks = extract_chunks(args.audio)
    except ValueError as e:
        logger.error(e)
        return

    predicted_class, probabilities = predict(model, chunks)

    print(f"\nPredicted class : {predicted_class}")
    print("Probabilities:")
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {cls:<16} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
