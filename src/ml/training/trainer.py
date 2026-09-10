"""PyTorch Model Training, Validation, and Early Stopping Pipeline.

Provides deterministic seed setting, training/validation loops, MSE loss optimization,
early stopping, history tracking, and model checkpoint management.
"""

import os
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch for reproducible training.

    Args:
        seed (int): Seed number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelTrainer:
    """Trainer class for PyTorch neural network regression models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        patience: int = 10,
    ):
        """Initialize trainer.

        Args:
            model (nn.Module): Neural network model.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            criterion (Optional[nn.Module]): Loss function (default MSELoss).
            device (str): Computing device ("cpu" or "cuda").
            patience (int): Early stopping patience epochs.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.device = device
        self.patience = patience

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            dataloader (DataLoader): Training DataLoader.

        Returns:
            float: Average training loss over epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(1, n_samples)

    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Run one validation epoch.

        Args:
            dataloader (DataLoader): Validation DataLoader.

        Returns:
            float: Average validation loss over epoch.
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

        return total_loss / max(1, n_samples)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        checkpoint_save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Fit model with early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Maximum training epochs.
            checkpoint_save_path (Optional[str]): File path to save best checkpoint.

        Returns:
            Dict[str, List[float]]: Training history metrics dictionary.
        """
        no_improve_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            curr_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(curr_lr)

            # Check improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                no_improve_counter = 0

                if checkpoint_save_path:
                    os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        checkpoint_save_path,
                    )
            else:
                no_improve_counter += 1

            if no_improve_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {self.best_epoch}, val_loss: {self.best_val_loss:.6f})")
                break

        # Restore best model weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history
