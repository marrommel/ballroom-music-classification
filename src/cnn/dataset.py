import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramImageDataset(Dataset):
    def __init__(self, image_patch_list, base_dir, train: bool = False):
        self.image_patch_list = image_patch_list
        self.base_dir = base_dir
        self.train = train

    def _augment(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment-style masking. Operates on a normalized 2D spectrogram."""
        H, W = spec.shape
        # Frequency masking (up to 2 bands)
        for _ in range(np.random.randint(0, 3)):
            f = np.random.randint(0, max(1, H // 8))
            f0 = np.random.randint(0, max(1, H - f))
            spec[f0:f0 + f, :] = 0.0
        # Time masking (up to 2 bands)
        for _ in range(np.random.randint(0, 3)):
            t = np.random.randint(0, max(1, W // 8))
            t0 = np.random.randint(0, max(1, W - t))
            spec[:, t0:t0 + t] = 0.0
        return spec

    def __len__(self):
        return len(self.image_patch_list)

    def __getitem__(self, idx):
        image_metadata = self.image_patch_list[idx]

        # Load 2D matrices (Grayscale Images)
        img_array_mel = np.load(image_metadata['view1_image_path'])
        img_array_cqt = np.load(image_metadata['view2_image_path'])

        if self.train:
            # Time-shift (roll) — dance rhythm is periodic, so wrap-around is valid
            shift = np.random.randint(0, img_array_mel.shape[1])
            img_array_mel = np.roll(img_array_mel, shift, axis=1)
            img_array_cqt = np.roll(img_array_cqt, shift, axis=1)
            #img_array_mel = self._augment(img_array_mel)
            #img_array_cqt = self._augment(img_array_cqt)

        # Pixel Intensity Normalization (Z-score standardization for image contrast)
        img_array_mel = (img_array_mel - np.mean(img_array_mel)) / (np.std(img_array_mel) + 1e-6)
        img_array_cqt = (img_array_cqt - np.mean(img_array_cqt)) / (np.std(img_array_cqt) + 1e-6)

        # Convert to Vision Tensors and add a Channel dimension: (Channels=1, Height, Width)
        img_tensor_v1 = torch.tensor(img_array_mel, dtype=torch.float32).unsqueeze(0)
        img_tensor_v2 = torch.tensor(img_array_cqt, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(image_metadata['label'], dtype=torch.long)

        return img_tensor_v1, img_tensor_v2, label

def parse_image_dataset(base_dir: str, visual_classes: list[str]) -> dict:
    """Parses the directory and groups image patches by their Parent Scene.

    Args:
        base_dir: Root directory containing ``mel/`` and ``cqt/`` subdirs.
        visual_classes: List of dance-class folder names (must match on-disk names).

    Returns:
        Dict mapping parent_scene_id -> list of image-patch metadata dicts.
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

    # parent_scene_id as list of image patch dicts
    scene_groups: dict = {}

    for label_idx, visual_class in enumerate(visual_classes):
        view1_dir = os.path.join(abs_base_dir, 'mel', visual_class)
        search_pattern = os.path.join(view1_dir, '*.npy')

        for view1_path in glob.glob(search_pattern):
            filename = os.path.basename(view1_path)

            # Extract parent scene ID: "Albums-Fire-03_chunk000_mel.npy" -> "Albums-Fire-03"
            parent_scene_id = filename.split('_chunk')[0]

            view2_filename = filename.replace('_mel.npy', '_cqt.npy')
            view2_path = os.path.join(abs_base_dir, 'cqt', visual_class, view2_filename)

            if not os.path.exists(view2_path):
                continue  # Skip if missing the matching visual view

            if parent_scene_id not in scene_groups:
                scene_groups[parent_scene_id] = []

            scene_groups[parent_scene_id].append({
                'view1_image_path': view1_path,
                'view2_image_path': view2_path,
                'label': label_idx,
                'parent_scene_id': parent_scene_id
            })

    return scene_groups