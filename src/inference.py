import argparse

import librosa
import numpy as np
import torch

from cnn.model import DualStreamVisionNet

DANCE_CLASSES = ['DiscoFox', 'ChaChaCha', 'Rumba', 'Jive', 'Quickstep', 'Tango', 'VienneseWaltz', 'Waltz']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "best_vision_model.pt"
SAMPLE_RATE = 22050
CHUNK_DURATION = 15  # seconds


def load_model(checkpoint_path: str) -> DualStreamVisionNet:
    """Load a trained DualStreamVisionNet from a checkpoint file.

    Args:
        checkpoint_path: Path to the trained state-dict file.

    Returns:
        The model in eval mode, ready for inference.
    """
    model = DualStreamVisionNet(num_classes=len(DANCE_CLASSES)).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _min_max_normalize(array: np.ndarray) -> np.ndarray:
    """Normalise a 2D array to [0, 1].

    Args:
        array: Input 2D numpy array.

    Returns:
        Normalised array, or the original if range is zero.
    """
    min_val, max_val = array.min(), array.max()
    if max_val - min_val == 0:
        return array
    return (array - min_val) / (max_val - min_val)


def generate_spectrograms(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an audio file and generate mel and CQT spectrograms from a 30-second middle chunk.

    Replicates the preprocessing pipeline used during training (min-max normalisation).
    Only the required chunk is decoded, avoiding loading the full file into memory.

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

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_normalized = _min_max_normalize(mel_db).astype(np.float32)

    # CQT
    cqt = librosa.cqt(y=y, sr=SAMPLE_RATE, hop_length=512, n_bins=84)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_normalized = _min_max_normalize(cqt_db).astype(np.float32)

    return mel_normalized, cqt_normalized


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a 2D spectrogram array to a model-ready tensor with Z-score normalisation.

    Args:
        array: 2D float32 numpy array.

    Returns:
        Float32 tensor of shape ``(1, 1, H, W)`` on ``DEVICE``.
    """
    array = (array - array.mean()) / (array.std() + 1e-6)
    return torch.tensor(array).unsqueeze(0).unsqueeze(0).to(DEVICE)


def predict(model: DualStreamVisionNet, mel: np.ndarray, cqt: np.ndarray) -> tuple[str, dict[str, float]]:
    """Run inference on a (mel, cqt) spectrogram pair.

    Args:
        model: A loaded DualStreamVisionNet in eval mode.
        mel: 2D mel spectrogram array.
        cqt: 2D CQT spectrogram array.

    Returns:
        Tuple of (predicted_class_name, {class_name: probability}).
    """
    img_v1 = _to_tensor(mel)
    img_v2 = _to_tensor(cqt)

    with torch.no_grad():
        logits = model(img_v1, img_v2)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = int(torch.tensor(probs).argmax())
    prob_map = {cls: round(p, 4) for cls, p in zip(DANCE_CLASSES, probs)}
    return DANCE_CLASSES[predicted_idx], prob_map


def main() -> None:
    """CLI entry point: classify the dance style of an audio file."""
    parser = argparse.ArgumentParser(description="Ballroom dance style classifier")
    parser.add_argument("audio", help="Path to the audio file to classify (.wav, .mp3, etc.)")
    args = parser.parse_args()

    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)

    print(f"Generating spectrograms from first {CHUNK_DURATION}s of: {args.audio}")
    mel, cqt = generate_spectrograms(args.audio)

    predicted_class, probabilities = predict(model, mel, cqt)

    print(f"\nPredicted class : {predicted_class}")
    print("Probabilities:")
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {cls:<16} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
