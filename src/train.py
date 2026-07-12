import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from cnn.dataset import parse_image_dataset, SpectrogramImageDataset
from cnn.model import DualSpectrogramClassificationModel

# --- Computer Vision Project Configurations ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DATASET_DIR = './visual_embeddings'
DANCE_CLASSES = ['DiscoFox', 'ChaChaCha', 'Rumba', 'Jive', 'Quickstep', 'Tango', 'VienneseWaltz', 'Waltz']
# 'NoDance',

TEST_SET_PERCENTAGE = 0.0
K_FOLDS = 5
BATCH_SIZE = 64

EPOCHS = 100
LR_PATIENCE = 3
LR_FACTOR = 0.75
EARLY_STOP_PATIENCE = 15
WARMUP_EPOCHS = 20


def main():
    print(f"Initializing Vision Pipeline on device: {DEVICE}")

    # Parse image dataset and group by Parent Scene (to prevent spatial leakage)
    scene_groups = parse_image_dataset(IMAGE_DATASET_DIR, DANCE_CLASSES)

    # Flatten the dictionary into one master list of all chunks
    all_image_patches = []
    for parent_scene_id, patches in scene_groups.items():
        all_image_patches.extend(patches)

    # 3. Extract exactly what sklearn needs: X (dummy), y (labels), and groups (scene IDs)
    X = np.zeros(len(all_image_patches))  # Dummy X, we only need the indices
    y = [patch['label'] for patch in all_image_patches]
    groups = [patch['parent_scene_id'] for patch in all_image_patches]

    print(f"Total Unique Scenes: {len(scene_groups)}")

    # Stratified Group K-Fold Cross Validation on the visual training set
    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    best_overall_val_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Starting Vision Fold {fold + 1}/{K_FOLDS} ---")

        # Map indices back to parent scene IDs
        train_image_patches = [all_image_patches[i] for i in train_idx]
        val_image_patches = [all_image_patches[i] for i in val_idx]

        # Create PyTorch DataLoaders for the Vision models
        train_loader = DataLoader(SpectrogramImageDataset(train_image_patches, IMAGE_DATASET_DIR, train=True),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(SpectrogramImageDataset(val_image_patches, IMAGE_DATASET_DIR, train=False),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Initialize CNN Model, Loss Function, and Optimizer
        model = DualSpectrogramClassificationModel(num_classes=len(DANCE_CLASSES)).to(DEVICE)
        # TODO: added label_smoothing to reduce the problem of Ballroom classes being genuinely confusable
        criterion = nn.CrossEntropyLoss() #label_smoothing=0.1

        optimizer = torch.optim.AdamW([
            {'params': model.mel_spectrograms_branch.parameters(), 'lr': 1e-5},
            {'params': model.cqt_spectrograms_branch.parameters(), 'lr': 1e-5},
            {'params': model.reduce_mlp_features.parameters(), 'lr': 5e-3},
            {'params': model.reduce_cqt_features.parameters(), 'lr': 5e-3},
            {'params': model.classification_head.parameters(), 'lr': 5e-3},
        ], weight_decay=1e-2)

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
            model.train()
            train_loss = 0.0
            correct_predictions = 0
            total_images = 0

            for mel_img, cqt_img, labels in train_loader:
                # Move image tensors to GPU
                mel_img, cqt_img, labels = mel_img.to(DEVICE), cqt_img.to(DEVICE), labels.to(DEVICE)

                # Apply MixUp
                #mel_img, cqt_img, targets_a, targets_b, lam = mixup_data(mel_img, cqt_img, labels, alpha=0.1)

                optimizer.zero_grad()
                logits = model(mel_img, cqt_img)
                #loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * mel_img.size(0)
                _, predicted = logits.max(1)
                total_images += labels.size(0)
                correct_predictions += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct_predictions / total_images

            # Validation Step
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for mel_img, cqt_img, labels in val_loader:
                    mel_img, cqt_img, labels = mel_img.to(DEVICE), cqt_img.to(DEVICE), labels.to(DEVICE)

                    logits = model(mel_img, cqt_img)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * mel_img.size(0)
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
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

def mixup_data(x1, x2, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    main()