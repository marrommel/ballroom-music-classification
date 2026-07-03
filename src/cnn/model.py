import torch
import torch.nn as nn
import timm


class DualStreamVisionNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.4):
        super().__init__()

        # Using timm to load a State-of-the-Art Vision model (MobileNetV4 Large).
        # in_chans=1 configures the first Conv2D layer for 1-channel Grayscale images.
        # num_classes=0 removes the classification head, acting purely as a visual feature extractor.
        self.vision_branch_v1 = timm.create_model('mobilenetv4_conv_large.e600_r384_in1k',
                                                  pretrained=True, in_chans=1, num_classes=0)

        self.vision_branch_v2 = timm.create_model('mobilenetv4_conv_large.e600_r384_in1k',
                                                  pretrained=True, in_chans=1, num_classes=0)

        # Determine visual embedding size. MobileNetV4 Large outputs 960 spatial features.
        # Concatenating two branches gives 960 * 2 = 1920 spatial features.
        visual_feature_dim = self.vision_branch_v1.num_features * 2

        # Custom Multi-Layer Perceptron (MLP) head for visual feature fusion
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(visual_feature_dim, 512),
            nn.BatchNorm1d(512),  # Stabilizes spatial feature distributions
            nn.GELU(),  # Non-linear activation for complex visual patterns
            nn.Dropout(p=dropout_rate / 2),
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