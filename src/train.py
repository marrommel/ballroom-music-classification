import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from cnn.dataset import parse_image_dataset, SpectrogramImageDataset
from cnn.model import DualStreamVisionNet

# --- Computer Vision Project Configurations ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DATASET_DIR = './visual_embeddings'
DANCE_CLASSES = ['DiscoFox', 'ChaChaCha', 'Rumba', 'Jive', 'Quickstep', 'Tango', 'VienneseWaltz', 'Waltz']

TEST_SET_PERCENTAGE = 0.0 
K_FOLDS = 5
BATCH_SIZE = 32

EPOCHS = 50
LR_PATIENCE = 3
LR_FACTOR = 0.75
EARLY_STOP_PATIENCE = 10


def main():
    print(f"Initializing Vision Pipeline on device: {DEVICE}")

    # 1. Parse image dataset and group by Parent Scene (to prevent spatial leakage)
    scene_groups = parse_image_dataset(IMAGE_DATASET_DIR, DANCE_CLASSES)
    unique_scenes = list(scene_groups.keys())

    # Extract labels per scene to maintain class balance during image splits
    scene_labels = [scene_groups[scene][0]['label'] for scene in unique_scenes]

    # 2. Configurable Train/Test Split (Holding out the Test set by parent scene ID)
    if TEST_SET_PERCENTAGE > 0:
        scenes_train_val, _ = train_test_split(
            unique_scenes,
            test_size=TEST_SET_PERCENTAGE,
            random_state=42,
            stratify=scene_labels
        )
    else:
        scenes_train_val = unique_scenes
    print(f"Total Unique Scenes: {len(unique_scenes)}")

    # 3. K-Fold Cross Validation on the visual training set
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    best_overall_val_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(scenes_train_val, scene_labels)):
        print(f"\n--- Starting Vision Fold {fold + 1}/{K_FOLDS} ---")

        # Map indices back to parent scene IDs
        fold_train_scenes = [scenes_train_val[i] for i in train_idx]
        fold_val_scenes = [scenes_train_val[i] for i in val_idx]

        # Flatten the image patches from the selected scenes into lists
        train_image_patches = [patch for scene in fold_train_scenes for patch in scene_groups[scene]]
        val_image_patches = [patch for scene in fold_val_scenes for patch in scene_groups[scene]]

        # Create PyTorch DataLoaders for the Vision models
        train_loader = DataLoader(SpectrogramImageDataset(train_image_patches, IMAGE_DATASET_DIR, train=False),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(SpectrogramImageDataset(val_image_patches, IMAGE_DATASET_DIR, train=False),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Initialize CNN Model, Loss Function, and Optimizer
        vision_model = DualStreamVisionNet(num_classes=len(DANCE_CLASSES)).to(DEVICE)
        # TODO: added label_smoothing to reduce the problem of Ballroom classes being genuinely confusable
        criterion = nn.CrossEntropyLoss() #label_smoothing=0.1
        optimizer = optim.AdamW(vision_model.parameters(), lr=1e-4, weight_decay=1e-2)

        # Reduce LR when val accuracy plateaus (fires before early stopping)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE
        )

        best_fold_val_acc = 0.0
        best_fold_state = None
        best_fold_epoch = 0
        epochs_without_improvement = 0

        # Training Loop over image batches
        for epoch in range(EPOCHS):
            vision_model.train()
            train_loss = 0.0
            correct_predictions = 0
            total_images = 0

            for img_v1, img_v2, labels in train_loader:
                # Move image tensors to GPU
                img_v1, img_v2, labels = img_v1.to(DEVICE), img_v2.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                logits = vision_model(img_v1, img_v2)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * img_v1.size(0)
                _, predicted = logits.max(1)
                total_images += labels.size(0)
                correct_predictions += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct_predictions / total_images

            # Validation Step
            vision_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for img_v1, img_v2, labels in val_loader:
                    img_v1, img_v2, labels = img_v1.to(DEVICE), img_v2.to(DEVICE), labels.to(DEVICE)

                    logits = vision_model(img_v1, img_v2)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * img_v1.size(0)
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * val_correct / val_total
            print(
                f"Fold {fold + 1} | Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss / total_images:.4f} Acc: {train_accuracy:.2f}% | Val Loss: {val_loss / val_total:.4f} Acc: {val_accuracy:.2f}%")

            # LR scheduler step (reduce LR on plateau)
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(val_accuracy)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"  ↓ LR reduced on plateau: {lr_before:.2e} → {lr_after:.2e}")

            # Save best epoch within this fold
            if val_accuracy > best_fold_val_acc:
                best_fold_val_acc = val_accuracy
                best_fold_state = {k: v.cpu().clone() for k, v in vision_model.state_dict().items()}
                best_fold_epoch = epoch + 1
                epochs_without_improvement = 0
                print(f"  ✓ New best for fold {fold + 1}: {val_accuracy:.2f}%")
            else:
                epochs_without_improvement += 1

            # Early stopping on sustained plateau (only after LR reduction had a chance)
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"  ⏹ Early stopping at epoch {epoch + 1}: no improvement for "
                    f"{EARLY_STOP_PATIENCE} epochs. Best epoch was {best_fold_epoch} "
                    f"(val acc {best_fold_val_acc:.2f}%).")
                break

        # Save best model across all folds
        if best_fold_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_fold_val_acc
            torch.save(best_fold_state, "best_vision_model.pt")
            print(f"  ★ New overall best saved (fold {fold + 1}, val acc {best_overall_val_acc:.2f}%)")

    print(f"\nTraining complete. Best overall val accuracy: {best_overall_val_acc:.2f}% → best_vision_model.pt")


if __name__ == '__main__':
    main()