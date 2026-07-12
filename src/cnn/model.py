import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file


def _load_grayscale_mobile_net_backbone(checkpoint_path: str) -> nn.Module:
    """Load a MobileNetV4 backbone from an RGB checkpoint, adapting it for grayscale input.

    Args:
        checkpoint_path: Path to the safetensors checkpoint file.

    Returns:
        A MobileNetV4 model configured for 1-channel input with no classification head.
    """
    # Load pretrained weights and drop the classification head
    state_dict = load_file(checkpoint_path)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

    # Average 3-channel (RGB) weights into 1-channel (grayscale)
    if "conv_stem.weight" in state_dict:
        state_dict["conv_stem.weight"] = state_dict["conv_stem.weight"].mean(dim=1, keepdim=True)

    # Load Mobile Net V4 small with its pretrained weights
    model = timm.create_model(
        'mobilenetv4_conv_small.e2400_r224_in1k',
        pretrained=False,
        in_chans=1,
        num_classes=0,
        drop_path_rate=0.4, # TODO: added stochastic depth, which is a strong regularizer
    )
    model.load_state_dict(state_dict, strict=False)
    return model


class DualSpectrogramClassificationModel(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.4):
        super().__init__()

        # Load backbones for mel and cqt classification branches
        pretrained_weights = r"src/mobilenetv4_conv_small_e2400_r224_in1k.safetensors"
        self.mel_spectrograms_branch = _load_grayscale_mobile_net_backbone(pretrained_weights)
        self.cqt_spectrograms_branch = _load_grayscale_mobile_net_backbone(pretrained_weights)

        # Determine the output feature size
        with torch.no_grad():
            self.mel_spectrograms_branch.eval()
            _dummy = torch.zeros(1, 1, 384, 384)
            _feat_dim = self.mel_spectrograms_branch(_dummy).shape[1]
            self.mel_spectrograms_branch.train()

        reduced_dim = 256
        self.reduce_mlp_features = nn.Sequential(
            nn.Linear(_feat_dim, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.GELU()
        )
        self.reduce_cqt_features = nn.Sequential(
            nn.Linear(_feat_dim, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.GELU()
        )

        # Custom Multi-Layer Perceptron (MLP) head for visual feature fusion
        # TODO: Use this with reduce_dim --> fused_dim = reduced_dim * 2
        fused_dim = _feat_dim * 2
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 1.5),
            nn.Linear(256, num_classes)
        )

    def set_backbone_frozen(self, frozen: bool):
        for param in self.mel_spectrograms_branch.parameters():
            param.requires_grad = not frozen
        for param in self.cqt_spectrograms_branch.parameters():
            param.requires_grad = not frozen

    def forward(self, mel_image, cqt_image):
        # Extract visual feature maps: Output Shape (Batch, Feature_Dim)
        spatial_features_mel = self.mel_spectrograms_branch(mel_image)
        spatial_features_cqt = self.cqt_spectrograms_branch(cqt_image)

        #spatial_features_mel = self.reduce_mlp_features(spatial_features_mel)
        #spatial_features_cqt = self.reduce_cqt_features(spatial_features_cqt)

        # Late Fusion: Concatenate the visual visual_embeddings from both image representations
        fused_visual_features = torch.cat((spatial_features_mel, spatial_features_cqt), dim=1)

        # Classify the fused visual patterns
        class_logits = self.classification_head(fused_visual_features)
        return class_logits