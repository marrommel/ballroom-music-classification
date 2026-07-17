import glob
import os

import numpy as np
import torch
import torchaudio.transforms as audio_transforms
from torch.utils.data import Dataset
from config import Config


class SpecDatasetEntry(Dataset):
    """PyTorch Dataset for spectrogram chunks with augmentation.

    Loads mel and CQT spectrograms with optional data augmentation for training.
    """

    def __init__(self, image_patch_list: list[dict], train: bool) -> None:
        """Initialize the dataset.

        Args:
            image_patch_list: List of metadata dicts with spectrogram paths and labels.
            train: Whether to apply data augmentation.
        """
        self.song_chunk_list = image_patch_list
        self.train = train

        self.config = Config()
        self.spec_types = self.config.spec_types

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.song_chunk_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a spectrogram pair and label by index.

        Args:
            idx: Index of the sample.

        Returns:
            Dict with spec_type keys mapping to tensors, plus a 'label' key.
        """
        song_metadata = self.song_chunk_list[idx]
        tensors: dict[str, torch.Tensor] = {}

        # Load spectrograms as grayscale images and preprocess them as tensors
        for spec_type in self.spec_types:
            spec_img = np.load(song_metadata[f"{spec_type}_image_path"])
            spec_tensor = torch.tensor(spec_img, dtype=torch.float32).unsqueeze(0)

            # Z-score normalization per spectrogram
            if  self.config.z_score_normalization_enabled:
                spec_tensor = (spec_tensor - spec_tensor.mean()) / (spec_tensor.std() + 1e-6)

            tensors[spec_type] = spec_tensor

        # apply selected augmentations for training data
        if self.train:
            for key, tensor in tensors.items():
                tensor = self._time_shift_wrap_around(tensor, self.config.time_shift_enabled)
                tensor = self._spec_augment(tensor,  self.config.spec_augment_enabled)
                tensors[key] = tensor

        # return the preprocessed spectrograms as tensors
        tensors['label'] = torch.tensor(song_metadata['label'], dtype=torch.long)
        return tensors

    @staticmethod
    def _time_shift_wrap_around(spec_img: torch.Tensor, enabled: bool) -> torch.Tensor:
        """Apply a time-shift with wrap around to the spectrogram.

        Args:
            spec_img: Spectrogram tensor to shift.
            enabled: If True, apply the time-shift; otherwise return unchanged.

        Returns:
            The shifted spectrogram tensor.
        """
        if not enabled:
            return spec_img

        # generate a random shift between 0 and the time dimension size
        shift = torch.randint(0, spec_img.shape[2], (1,)).item()

        # apply the shift to every spectrogram
        return torch.roll(spec_img, shifts=shift, dims=2)

    @staticmethod
    def _spec_augment(
            spec: torch.Tensor,
            enabled: bool,
            freq_mask_param: int = 15,
            time_mask_param: int = 60,
    ) -> torch.Tensor:
        """Apply SpecAugment (Frequency and Time masking) to spectrogram.

        Args:
            spec: Input spectrogram tensor.
            enabled: If True, apply augmentation; otherwise return unchanged.
            freq_mask_param: Maximum width of the frequency mask.
            time_mask_param: Maximum width of the time mask.

        Returns:
            The augmented spectrogram tensor.
        """
        if not enabled:
            return spec

        # Create the transforms
        freq_masking = audio_transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        time_masking = audio_transforms.TimeMasking(time_mask_param=time_mask_param)

        # Apply the transforms
        augmented_tensor = time_masking(freq_masking(spec))

        return augmented_tensor


class SpecDatasetLoader:
    """Loads spectrogram dataset from disk organized by dance class.

    Collects spectrogram files across multiple types and organizes them by song.
    """

    def __init__(self, spec_types: list[str]) -> None:
        """Initialize the dataset loader.

        Args:
            spec_types: List of spectrogram type names (e.g., 'mel', 'cqt').
        """
        config = Config()
        self.base_dir = config.data_base_dir
        self.dance_classes = config.dance_classes
        self.spec_types = spec_types

    def load_dataset(self) -> dict:
        """Load all spectrograms and organize by song.

        Returns:
            Dict mapping parent_song names to lists of metadata dicts with paths and labels.
        """
        self._print_data_statistics()

        # Use the first spectrogram type as an anchor to iterate filenames
        first_spec_type = self.spec_types[0]
        first_spec_root = os.path.join(self.base_dir, first_spec_type)

        # Iterate over all dance class directories
        song_groups: dict = {}
        for label_idx, dance_class in enumerate(self.dance_classes):
            first_spec_dir = os.path.join(first_spec_root, dance_class)
            search_pattern = os.path.join(first_spec_dir, '*.npy')

            # Iterate over all ´*.npy´ files a dance directory
            for first_spec_path in glob.glob(search_pattern):
                filename = os.path.basename(first_spec_path)
                parent_song = filename.split('_chunk')[0]

                # Build metadata for all requested spec types
                metadata = {'label': label_idx, 'parent_song': parent_song}
                skip = False

                # Iterate over all spectrogram types
                for spec_type in self.spec_types:
                    spec_filename = filename.replace(first_spec_type, spec_type)
                    spec_path = os.path.join(self.base_dir, spec_type, dance_class, spec_filename)

                    # Skip if the assumes spectrogram does not exist
                    if not os.path.exists(spec_path):
                        skip = True
                        break

                    # Add another spec path to the metadata
                    metadata[f"{spec_type}_image_path"] = spec_path

                # Skip if data did not exist for all spec types
                if skip:
                    continue

                # Add the metadata to the result dict
                if parent_song not in song_groups:
                    song_groups[parent_song] = []
                song_groups[parent_song].append(metadata)

        return song_groups

    def _print_data_statistics(self) -> None:
        """Print statistics about available dataset files."""
        for spec_type in self.spec_types:
            data_root = os.path.join(self.base_dir, spec_type)
            if os.path.isdir(data_root):
                subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
                total_files = sum(
                    len(glob.glob(os.path.join(data_root, sd, '*.npy'))) for sd in subdirs
                )
                print(f"[dataset]   {spec_type}/  →  {len(subdirs)} class folder(s), {total_files} .npy file(s)")
            else:
                print(f"[dataset]   WARNING: '{data_root}' does not exist")

