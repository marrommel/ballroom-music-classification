import argparse
import logging
import math
import sys

import librosa
import numpy as np
import torch

from src.cnn.dataset import SpecDatasetEntry
from src.cnn.model import MultiSpectrogramClassificationModel
from src.config import Config
from src.helpers.spectrograms import compute_spectrograms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEVICE = 'cpu'
SAMPLE_RATE = 22050

config = Config()
CHUNK_DURATION = config.chunk_duration
INFERENCE_DURATION = config.inference_duration
MIN_DURATION = config.min_chunk_duration


def load_model(checkpoint_path: str) -> MultiSpectrogramClassificationModel:
    """Load a trained MultiSpectrogramClassificationModel from a checkpoint file.

    Args:
        checkpoint_path: Path to the trained state-dict file.

    Returns:
        The model in eval mode, ready for inference.
    """
    model = MultiSpectrogramClassificationModel().to(DEVICE)
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


def extract_chunks(audio_path: str, spec_types: list[str]) -> list[dict[str, np.ndarray]]:
    """Load audio, center a window of *inference_duration* seconds, and return
    spectrogram dicts for each CHUNK_DURATION-sized slice with 50% overlap.

    Args:
        audio_path: Path to the audio file.
        spec_types: List of spectrogram types to compute (e.g. ['mel', 'cqt']).

    Returns:
        List of {spec_type: array} dicts — one entry per chunk.

    Raises:
        ValueError: if the audio or requested duration is below MIN_DURATION.
    """
    total_duration = math.floor(librosa.get_duration(path=audio_path))
    if not validate_durations(total_duration):
        return []

    inference_duration = min(INFERENCE_DURATION, total_duration)
    offset = config.inference_offset
    offset = min(offset, total_duration-INFERENCE_DURATION)
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, offset=offset, duration=inference_duration)

    samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)

    # For short inference_duration return only one chunk
    if len(y) < samples_per_chunk:
        return [compute_spectrograms(y, SAMPLE_RATE, spec_types)]

    # Sliding-window chunking with 50% overlap
    hop_samples = samples_per_chunk // 2
    total_chunks = (len(y) - samples_per_chunk) // hop_samples + 1
    return [
        compute_spectrograms(y[i * hop_samples: i * hop_samples + samples_per_chunk], SAMPLE_RATE, spec_types)
        for i in range(total_chunks)
    ]


def predict(
    model: MultiSpectrogramClassificationModel,
    chunks: list[dict[str, np.ndarray]]
) -> tuple[str, dict[str, float]]:
    """Run inference on a list of spectrogram dicts.

    Args:
        model: A loaded MultiSpectrogramClassificationModel in eval mode.
        chunks: List of {spec_type: array} dicts, one per chunk.

    Returns:
        Tuple of (predicted_class_name, {class_name: probability}).
    """
    # Preprocess and batch all chunks together: {spec_type: (N, C, H, W)}
    batched_specs = {
        s: torch.stack([
            SpecDatasetEntry.preprocess_spectrogram(chunk[s]) for chunk in chunks
        ]).to(DEVICE)
        for s in config.spec_types
    }

    with torch.no_grad():
        logits = model(batched_specs)  # (N, num_classes)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Entropy per chunk (lower = more confident)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (N,)

        # Invert entropy into weights: low entropy -> high weight
        temperature = 1  # >1 sharpens weighting, <1 softens it
        weights = torch.softmax(-entropy * temperature, dim=0)

        # Confidence-weighted average of probabilities
        final_probs = torch.sum(probs * weights.unsqueeze(1), dim=0)  # (num_classes,)

    final_probs = final_probs.cpu().tolist()
    predicted_idx = int(np.argmax(final_probs))
    prob_map = {cls: round(p, 4) for cls, p in zip(config.dance_classes, final_probs)}
    return config.dance_classes[predicted_idx], prob_map


def main() -> None:
    """CLI entry point: classify the dance style of an audio file."""
    parser = argparse.ArgumentParser(description="Ballroom dance style classifier")
    parser.add_argument("audio", help="Path to the audio file to classify (.wav, .mp3, etc.)")
    args = parser.parse_args()

    print(f"Loading model from: {config.inference_model_weights}")
    model = load_model(config.inference_model_weights)

    print(f"Generating spectrograms ({', '.join(config.spec_types)}) for: {args.audio}")
    try:
        chunks = extract_chunks(args.audio, config.spec_types)
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

