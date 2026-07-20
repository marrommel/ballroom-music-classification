import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from cnn.dataset import SpecDatasetEntry, SpecDatasetLoader
from cnn.model import MultiSpectrogramClassificationModel
from config import Config
from inference import DEVICE


class TrainingHandler:
	"""Handles model training with k-fold cross-validation and early stopping."""

	def __init__(self) -> None:
		"""Initialize training handler with configuration and device setup."""
		self.config = Config()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Per-fold state variables
		self.model: MultiSpectrogramClassificationModel
		self.optimizer: torch.optim.Optimizer
		self.criterion: nn.Module
		self.scheduler = None

		# Per-fold best tracking
		self._best_fold_state: dict
		self._best_fold_val_acc: float = 0.0
		self._best_fold_epoch: int = 0
		self._epochs_without_improvement: int = 0

	def train(self) -> None:
		"""Execute k-fold cross-validation training loop."""
		print(f"Initializing Model Training on device: {self.device}")
		all_chunks, X, y, groups = self._prepare_data()
		kfold = StratifiedGroupKFold(
			n_splits=self.config.k_folds, shuffle=True, random_state=self.config.k_fold_rand_id
		)

		# Iterate iver all folds
		best_overall_val_acc = 0.0
		for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups)):
			if fold >= self.config.max_fold: continue
			print(f"\n--- Starting Fold {fold + 1}/{self.config.k_folds} ---")

			# prepare variables for a new fold
			train_loader, val_loader = self._build_loaders(all_chunks, train_idx, val_idx)
			self._setup_fold()

			# start training and validation for the given number of epochs
			best_fold_val_acc = self._start_training(fold, train_loader, val_loader)

			# store model weights if the fold achieves a new best accuracy
			if best_fold_val_acc > best_overall_val_acc:
				best_overall_val_acc = best_fold_val_acc
				torch.save(self._best_fold_state, "best_model.pt")
				print(f"  ★ New overall best saved (fold {fold + 1}, val acc {best_overall_val_acc:.2f}%)")

		print(f"\nTraining complete. Best overall val accuracy: "
			f"{best_overall_val_acc:.2f}% → best_model.pt")

	def _prepare_data(self) -> tuple[list, np.ndarray, list, list]:
		"""Load and prepare dataset chunks grouped by parent song.

		Returns:
			Tuple of (all_chunks, dummy_X, labels, parent_song_groups).
		"""
		# load a spectrogram image chunks grouped by their parent song
		song_spec_data = SpecDatasetLoader(self.config.spec_types).load_dataset()

		# flatten the grouped chunks into a single array
		all_chunks = []
		for chunks in song_spec_data.values():
			all_chunks.extend(chunks)

		X = np.zeros(len(all_chunks))  # Dummy — only indices matter for splitting
		y = [chunk['label'] for chunk in all_chunks]
		groups = [chunk['parent_song'] for chunk in all_chunks]

		print(f"Total Unique Songs: {len(song_spec_data)}")
		return all_chunks, X, y, groups

	def _build_loaders(self, all_chunks: list, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[DataLoader, DataLoader]:
		"""Create train and validation data loaders.

		Args:
			all_chunks: List of all dataset chunks.
			train_idx: Indices for training set.
			val_idx: Indices for validation set.

		Returns:
			Tuple of (train_loader, val_loader).
		"""
		# split chunks into train and validation
		train_chunks = [all_chunks[i] for i in train_idx]
		val_chunks = [all_chunks[i] for i in val_idx]

		# instantiate data loaders for both datasets
		train_loader = DataLoader(
			SpecDatasetEntry(train_chunks, train=self.config.train_data_augmentation),
			batch_size=self.config.batch_size, shuffle=True, num_workers=4,
		)
		val_loader = DataLoader(
			SpecDatasetEntry(val_chunks, train=False),
			batch_size=self.config.batch_size, shuffle=False, num_workers=4,
		)
		return train_loader, val_loader

	def _setup_fold(self) -> None:
		"""Initialize model, optimizer, and scheduler for a new fold."""
		# Reset model and loss function for the new fold
		self.model = MultiSpectrogramClassificationModel().to(self.device)
		weights = torch.tensor([0.7, 1.2, 1.0, 1.2, 1.0, 1.3, 1.0, 1.0]).to(DEVICE)
		self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.config.label_smoothing)

		# Configure optimizer for backbone branches, feature reduction and classification head
		self.optimizer = torch.optim.AdamW([
			*[{'params': self.model.branches[s].parameters(), 'lr': self.config.lr_backbone}
			  for s in self.model.branches.keys()],
			*[{'params': self.model.reduce_mlps[s].parameters(), 'lr': self.config.lr_feature_reduction}
			  for s in self.model.reduce_mlps.keys()],
			{'params': self.model.classification_head.parameters(), 'lr': self.config.lr_head},
		], weight_decay=self.config.weight_decay)

		# Configure LR schedular mode
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode=self.config.lr_scheduler_mode,
			factor=self.config.lr_factor,
			patience=self.config.lr_patience,
		)

		# Reset best tracking
		self._best_fold_val_acc = 0.0
		self._best_fold_state = None
		self._best_fold_epoch = 0
		self._epochs_without_improvement = 0

	def _start_training(self, fold: int, train_loader: DataLoader, val_loader: DataLoader) -> float:
		"""Run training loop for a fold with early stopping.

		Args:
			fold: Fold number.
			train_loader: Training data loader.
			val_loader: Validation data loader.

		Returns:
			Best validation accuracy achieved in the fold.
		"""
		# Iterate over the given number of epochs
		for epoch in range(self.config.epochs):
			if epoch == 0:
				self.model.freeze_backbone(True)
			elif epoch == 5:
				self.model.freeze_backbone(False)

			# Run epoch on train and validation data
			train_loss, train_acc = self._run_single_epoch(train_loader, train=True)
			val_loss, val_acc = self._run_single_epoch(val_loader, train=False)

			print(
				f"Fold {fold + 1} | Epoch {epoch + 1}/{self.config.epochs} | "
				f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
				f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
			)

			# update the LR schedular
			self._step_scheduler(val_loss, val_acc)

			# check if the current epoch improved and if early stopping has occurred
			improved = self._update_best(val_acc, epoch, fold)
			if not improved and self._should_early_stop(epoch):
				break

		return self._best_fold_val_acc

	def _run_single_epoch(self, data_loader: DataLoader, train: bool) -> tuple[float, float]:
		"""Run single epoch on train or validation data.

		Args:
			data_loader: Data loader for epoch.
			train: Whether to run in training mode.

		Returns:
			Tuple of (average_loss, accuracy_percent).
		"""
		total_loss = 0.0
		correct = 0
		total = 0

		# Configure the correct mode to run the model
		self.model.train() if train else self.model.eval()

		# Disable gradient calculation for validation
		context = contextlib.nullcontext() if train else torch.no_grad()
		with context:

			# Iterate over the dataset of spectrograms and labels
			for data in data_loader:
				labels = data['label'].to(self.device)
				specs = {s: data[s].to(self.device) for s in self.config.spec_types}

				# If enabled, apply mixup durin training
				if train and self.config.train_data_mixup:
					specs, targets_a, targets_b, ratio = self.mixup_data(specs, labels)

				# Reset gradients during training
				if train:
					self.optimizer.zero_grad()

				# calculate model output and loss
				logits = self.model(specs)
				if train and self.config.train_data_mixup:
					loss = self.mixup_criterion(self.criterion, logits, targets_a, targets_b, ratio)
				else:
					loss = self.criterion(logits, labels)

				# Perform backpropagation during training
				if train:
					loss.backward()
					if self.config.clip_weights:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.optimizer.step()

				# Calculate training and validation metrics
				data_size = next(iter(specs.values())).size(0)
				total_loss += loss.item() * data_size
				_, predicted = logits.max(1)
				total += labels.size(0)

				# Use original labels for accuracy; targets_a are the unshuffled originals
				ref_labels = targets_a if (train and self.config.train_data_mixup) else labels
				correct += predicted.eq(ref_labels).sum().item()

		return total_loss / total, 100.0 * correct / total

	def _step_scheduler(self, val_loss: float, val_acc: float) -> None:
		"""Update learning rate scheduler and log changes.

		Args:
			val_loss: Validation loss value.
			val_acc: Validation accuracy value.
		"""
		# use the correct metric for LR scheduler
		metric = val_acc if self.config.lr_scheduler_mode == "max" else val_loss

		# update LR with scheduler
		lr_before = self.optimizer.param_groups[0]['lr']
		self.scheduler.step(metric)
		lr_after = self.optimizer.param_groups[0]['lr']

		if lr_after < lr_before:
			print(f"  ↓ LR reduced on plateau: {lr_before:.2e} → {lr_after:.2e}")

	def _update_best(self, val_acc: float, epoch: int, fold: int) -> bool:
		"""Save fold state if validation accuracy improved.

		Args:
			val_acc: Current validation accuracy.
			epoch: Current epoch number.
			fold: Current fold number.

		Returns:
			True if validation accuracy improved, False otherwise.
		"""
		if val_acc > self._best_fold_val_acc:
			self._best_fold_val_acc = val_acc
			self._best_fold_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
			self._best_fold_epoch = epoch + 1
			self._epochs_without_improvement = 0
			print(f"  ✓ New best for fold {fold + 1}: {val_acc:.2f}%")
			return True
		self._epochs_without_improvement += 1
		return False

	def _should_early_stop(self, epoch: int) -> bool:
		"""Check if early stopping criteria is met.

		Args:
			epoch: Current epoch number.

		Returns:
			True if early stopping should be triggered, False otherwise.
		"""
		if self._epochs_without_improvement >= self.config.early_stop_patience:
			print(
				f"  ⏹ Early stopping at epoch {epoch + 1}: no improvement for "
				f"{self.config.early_stop_patience} epochs. Best epoch was {self._best_fold_epoch} "
				f"(val acc {self._best_fold_val_acc:.2f}%)."
			)
			return True
		return False

	@staticmethod
	def mixup_data(specs: dict[str, torch.Tensor], y: torch.Tensor, alpha: float = 0.2) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
		"""Apply mixup augmentation to spectrogram batch.

		Args:
			specs: Dictionary of spectrogram tensors.
			y: Target labels.
			alpha: Beta distribution parameter for mixing ratio.

		Returns:
			Tuple of (mixed_specs, original_targets, shuffled_targets, mixing_ratio).
		"""
		ratio = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
		index = torch.randperm(y.size(0)).to(y.device)
		mixed_specs = {s: ratio * x + (1 - ratio) * x[index] for s, x in specs.items()}
		return mixed_specs, y, y[index], ratio

	@staticmethod
	def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, ratio: float) -> torch.Tensor:
		"""Compute mixup loss combining two label targets.

		Args:
			criterion: Loss function.
			pred: Model predictions.
			y_a: First target labels.
			y_b: Second target labels.
			ratio: Mixing ratio for loss combination.

		Returns:
			Combined loss value.
		"""
		return ratio * criterion(pred, y_a) + (1 - ratio) * criterion(pred, y_b)


if __name__ == '__main__':
	TrainingHandler().train()

