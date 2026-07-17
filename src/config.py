from dataclasses import dataclass, field


@dataclass
class Config:
    # Cross-validation
    max_fold: int = 2
    k_folds: int = 5
    batch_size: int = 32
    k_fold_rand_id: int = 39

    # Training
    epochs: int = 100
    early_stop_patience: int = 20

    # Optimizer
    weight_decay: float = 5e-2
    lr_backbone: float = 1.5e-5
    lr_feature_reduction: float = 5e-4
    lr_head: float = 5e-4

    # LR Scheduler
    lr_patience: int = 3
    lr_factor: float = 0.75
    lr_scheduler_mode: str = "min"

    # Augmentation
    train_data_augmentation: bool = True
    time_shift_enabled = False
    spec_augment_enabled = True
    train_data_mixup: bool = False
    mixup_rate: float = 0.2

    # Regularization
    clip_weights: bool = False
    label_smoothing: float = 0  # TODO: label_smoothing=0.15
    head_dropout_rate: float = 0.4
    backbone_drop_path_rate: float = 0.15
    reduced_dim = 256

    # Data
    spec_types: list[str] = field(default_factory=lambda: ["mel", "temp"])
    chunk_duration: int = 15
    min_chunk_duration: int = 3
    data_base_dir = "./assets/image_embeddings"
    dance_classes: list[str] = field(default_factory=lambda: [
        'DiscoFox', 'ChaChaCha', 'Rumba', 'Jive', 'Quickstep', 'Tango', 'VienneseWaltz', 'Waltz'
    ])

    # Model
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained_weights: str = "./assets/mobilenetv4_conv_small_e2400_r224_in1k.safetensors"
    z_score_normalization_enabled = False


    # Inference
    inference_duration: int = 30
    inference_model_weights = "best_model.pt"
