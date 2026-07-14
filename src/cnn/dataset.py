import glob
import os

import numpy as np
import torch
import torchaudio.transforms as audio_transforms
from torch.utils.data import Dataset


class SpectrogramImageDataset(Dataset):
    def __init__(self, image_patch_list, base_dir, train: bool = False):
        self.image_patch_list = image_patch_list
        self.base_dir = base_dir
        self.train = train

    def __len__(self):
        return len(self.image_patch_list)

    def __getitem__(self, idx):
        image_metadata = self.image_patch_list[idx]

        # Load 2D matrices (Grayscale Images)
        img_array_mel = np.load(image_metadata['mel_image_path'])
        img_array_cqt = np.load(image_metadata['cqt_image_path'])

        # Convert to tensors with channel dimension: (Channels=1, Height, Width)
        img_tensor_mel = torch.tensor(img_array_mel, dtype=torch.float32).unsqueeze(0)
        img_tensor_cqt = torch.tensor(img_array_cqt, dtype=torch.float32).unsqueeze(0)

        # 2. Pixel Intensity Normalization (Z-score standardization)
        #img_tensor_mel = (img_tensor_mel - img_tensor_mel.mean()) / (img_tensor_mel.std() + 1e-6)
        #img_tensor_cqt = (img_tensor_cqt - img_tensor_cqt.mean()) / (img_tensor_cqt.std() + 1e-6)

        if self.train:
            # Time-shift (roll) — dance rhythm is periodic, so wrap-around is valid
            shift = torch.randint(0, img_tensor_mel.shape[2], (1,)).item()
            img_tensor_mel = torch.roll(img_tensor_mel, shifts=shift, dims=2)
            img_tensor_cqt = torch.roll(img_tensor_cqt, shifts=shift, dims=2)
            img_tensor_mel = self._spec_augment(img_tensor_mel)
            img_tensor_cqt = self._spec_augment(img_tensor_cqt)

        # return the preprocessed spectrograms as tensors
        label = torch.tensor(image_metadata['label'], dtype=torch.long)
        return img_tensor_mel, img_tensor_cqt, label

    @staticmethod
    def _spec_augment(
            spec: torch.Tensor,
            freq_mask_param: int = 15,
            time_mask_param: int = 60
    ) -> torch.Tensor:
        """
        Applies SpecAugment (Frequency and Time masking) to a spectrogram.

        Args:
            spec (torch.Tensor): The input spectrogram tensor.
            freq_mask_param (int): Maximum width of the frequency mask.
            time_mask_param (int): Maximum width of the time mask.

        Returns:
            torch.Tensor: The augmented spectrogram.
        """
        # Create the transforms
        freq_masking = audio_transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        time_masking = audio_transforms.TimeMasking(time_mask_param=time_mask_param)

        # Apply the transforms
        augmented_tensor = time_masking(freq_masking(spec))

        return augmented_tensor

def parse_image_dataset(base_dir: str, dance_classes: list[str]) -> dict:
    """Parses the directory and groups image patches by their Parent Scene.

    Args:
        base_dir: Root directory containing ``mel/`` and ``cqt/`` subdirs.
        dance_classes: List of dance-class folder names (must match on-disk names).

    Returns:
        Dict mapping parent_song -> list of image-patch metadata dicts.
    """
    abs_base_dir = os.path.abspath(base_dir)
    mel_root = os.path.join(abs_base_dir, 'mel')
    cqt_root = os.path.join(abs_base_dir, 'cqt')

    print(f"[dataset] base_dir resolved to: {abs_base_dir}")
    for view_root, view_name in ((mel_root, 'mel'), (cqt_root, 'cqt')):
        if os.path.isdir(view_root):
            subdirs = [d for d in os.listdir(view_root) if os.path.isdir(os.path.join(view_root, d))]
            total_files = sum(
                len(glob.glob(os.path.join(view_root, sd, '*.npy'))) for sd in subdirs
            )
            print(f"[dataset]   {view_name}/  →  {len(subdirs)} class folder(s), {total_files} .npy file(s)")
        else:
            print(f"[dataset]   WARNING: '{view_root}' does not exist")

    # parent_song as list of image patch dicts
    song_groups: dict = {}

    for label_idx, dance_class in enumerate(dance_classes):
        mel_dir = os.path.join(abs_base_dir, 'mel', dance_class)
        search_pattern = os.path.join(mel_dir, '*.npy')

        for mel_img_path in glob.glob(search_pattern):
            filename = os.path.basename(mel_img_path)

            # extract the song name of the spectrogram chunk
            parent_song = filename.split('_chunk')[0]

            # build the cqt image path for the corresponding mel image
            cqt_filename = filename.replace('_mel.npy', '_cqt.npy')
            cqt_img_path = os.path.join(abs_base_dir, 'cqt', dance_class, cqt_filename)

            # skip if the matching cqt image is missing
            if not os.path.exists(cqt_img_path):
                continue

            if parent_song not in song_groups:
                song_groups[parent_song] = []

            song_groups[parent_song].append({
                'mel_image_path': mel_img_path,
                'cqt_image_path': cqt_img_path,
                'label': label_idx,
                'parent_song': parent_song
            })

    return song_groups