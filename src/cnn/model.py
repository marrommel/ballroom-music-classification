import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file
from timm.layers import drop_path


def _load_grayscale_backbone(checkpoint_path: str) -> nn.Module:
    """Load a MobileNetV4 backbone from an RGB checkpoint, adapting it for grayscale input.

    Removes the classifier head and converts the 3-channel conv_stem weights to
    1-channel by averaging across the channel dimension.

    Args:
        checkpoint_path: Path to the safetensors checkpoint file.

    Returns:
        A MobileNetV4 model configured for 1-channel input with no classification head.
    """
    state_dict = load_file(checkpoint_path)

    # Drop the classification head — model is built with num_classes=0
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

    # Adapt conv_stem from 3-channel (RGB) to 1-channel (grayscale) by averaging
    if "conv_stem.weight" in state_dict:
        state_dict["conv_stem.weight"] = state_dict["conv_stem.weight"].mean(dim=1, keepdim=True)

    model = timm.create_model(
        #'mobilenetv4_conv_large.e600_r384_in1k',
        'mobilenetv4_conv_small.e2400_r224_in1k',
        pretrained=False,
        in_chans=1,
        num_classes=0,
    )
    model.load_state_dict(state_dict, strict=False)
    return model


class DualStreamVisionNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.4):
        super().__init__()

        #pretrained_weights = r"src/mobilenetv4_conv_large_e600_r384_in1k.safetensors"
        pretrained_weights = r"src/mobilenetv4_conv_small_e2400_r224_in1k.safetensors"

        # Using timm to load a State-of-the-Art Vision model (MobileNetV4 Large).
        # in_chans=1 configures the first Conv2D layer for 1-channel Grayscale images.
        # num_classes=0 removes the classification head, acting purely as a visual feature extractor.
        self.vision_branch_v1 = _load_grayscale_backbone(pretrained_weights)
        self.vision_branch_v2 = _load_grayscale_backbone(pretrained_weights)

        # Infer the true feature dim with a dummy forward pass, since num_features
        # does not reliably reflect the pooled output size when num_classes=0.
        with torch.no_grad():
            self.vision_branch_v1.eval()
            _dummy = torch.zeros(1, 1, 384, 384)
            _feat_dim = self.vision_branch_v1(_dummy).shape[1]
            self.vision_branch_v1.train()
        visual_feature_dim = _feat_dim * 2

        # Custom Multi-Layer Perceptron (MLP) head for visual feature fusion
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(visual_feature_dim, 512),
            nn.BatchNorm1d(512),  # Stabilizes spatial feature distributions
            nn.GELU(),  # Non-linear activation for complex visual patterns
            nn.Dropout(p=dropout_rate / 1.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image_view1, image_view2):
        # Extract visual feature maps: Output Shape (Batch, Feature_Dim)
        spatial_features_v1 = self.vision_branch_v1(image_view1)
        spatial_features_v2 = self.vision_branch_v2(image_view2)

        # Late Fusion: Concatenate the visual visual_embeddings from both image representations
        fused_visual_features = torch.cat((spatial_features_v1, spatial_features_v2), dim=1)

        # Classify the fused visual patterns
        class_logits = self.classification_head(fused_visual_features)
        return class_logits