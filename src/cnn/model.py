import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file

from config import Config


class MultiSpectrogramClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        spec_types = self.config.spec_types

        # Create branches dynamically for each spectrogram type
        self.branches = nn.ModuleDict()
        self.reduce_mlps = nn.ModuleDict()

        # Load backbone for all branches
        for spec_type in spec_types:
            self.branches[spec_type] = self._load_grayscale_mobile_net_backbone()

        # Determine output feature dimension from first branch
        with torch.no_grad():
            first_branch = next(iter(self.branches.values()))
            first_branch.eval()
            _dummy = torch.zeros(1, 1, 384, 384)
            _feat_dim = first_branch(_dummy).shape[1]
            first_branch.train()

        # Create reduce MLPs for each branch
        for spec_type in spec_types:
            self.reduce_mlps[spec_type] = self._create_reduce_mlp(_feat_dim, self.config.reduced_dim)

        # Create the classification head based on the number of branches
        fused_dim = self.config.reduced_dim * len(spec_types)
        self.classification_head = self._create_classification_head(fused_dim)

    def freeze_backbone(self, frozen: bool):
        for branch in self.branches.values():
            for param in branch.parameters():
                param.requires_grad = not frozen

    def forward(self, spectrograms: dict[str, torch.Tensor]):
        """Forward pass with variable number of spectrogram inputs.

        Args:
            spectrograms: Dictionary mapping spec_type to image tensor
        """
        features = []

        # iterate over all backbone branches
        for spec_type in self.branches.keys():
            spatial_features = self.branches[spec_type](spectrograms[spec_type])
            reduced_features = self.reduce_mlps[spec_type](spatial_features)
            features.append(reduced_features)

        # Fuse all branch outputs and pass them to the classification head
        fused_features = torch.cat(features, dim=1)
        return self.classification_head(fused_features)

    def _load_grayscale_mobile_net_backbone(self) -> nn.Module:
        """Load a MobileNetV4 backbone from an RGB checkpoint, adapting it for grayscale input.

        Returns:
            A MobileNetV4 model configured for 1-channel input with no classification head.
        """
        # Load pretrained weights and drop the classification head
        state_dict = load_file(self.config.pretrained_weights)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

        # Average 3-channel (RGB) weights into 1-channel (grayscale)
        if "conv_stem.weight" in state_dict:
            state_dict["conv_stem.weight"] = state_dict["conv_stem.weight"].mean(dim=1, keepdim=True)

        # Load Mobile Net V4 small with its pretrained weights
        model = timm.create_model(
            self.config.model_name,
            pretrained=False,
            in_chans=1,
            num_classes=0,
            drop_path_rate=self.config.backbone_drop_path_rate,
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _create_reduce_mlp(feat_dim: int, reduced_dim:int) -> nn.Module:
        """Create an MLP for feature reduction."""
        return nn.Sequential(
            nn.Linear(feat_dim, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.GELU()
        )

    def _create_classification_head(self, input_dim: int) -> nn.Module:
        """Create a classification head for the given input dimension."""
        dropout_rate = self.config.head_dropout_rate
        num_classes = len(self.config.dance_classes)

        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            # OR nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 1.5),
            nn.Linear(256, num_classes),
        )