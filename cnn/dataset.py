import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramImageDataset(Dataset):
    def __init__(self, image_patch_list, base_dir):
        """
        image_patch_list: List of dictionaries containing image paths and labels for this specific fold
        """
        self.image_patch_list = image_patch_list
        self.base_dir = base_dir

    def __len__(self):
        return len(self.image_patch_list)

    def __getitem__(self, idx):
        image_metadata = self.image_patch_list[idx]

        # Load 2D matrices (Grayscale Images)
        img_array_v1 = np.load(image_metadata['view1_image_path'])  # Formerly Mel
        img_array_v2 = np.load(image_metadata['view2_image_path'])  # Formerly CQT

        # Pixel Intensity Normalization (Z-score standardization for image contrast)
        img_array_v1 = (img_array_v1 - np.mean(img_array_v1)) / (np.std(img_array_v1) + 1e-6)
        img_array_v2 = (img_array_v2 - np.mean(img_array_v2)) / (np.std(img_array_v2) + 1e-6)

        # Convert to Vision Tensors and add a Channel dimension: (Channels=1, Height, Width)
        img_tensor_v1 = torch.tensor(img_array_v1, dtype=torch.float32).unsqueeze(0)
        img_tensor_v2 = torch.tensor(img_array_v2, dtype=torch.float32).unsqueeze(0)
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