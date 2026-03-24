"""
trainer.py
----------
Training loop for the macro-conditional normalizing flow.

Optimizes the negative log-likelihood:
    L(θ) = -E[log p(X_t | h_t)]

where h_t is produced by the TFT encoder and log p is computed by the MAF.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.flow_model import ConditionalNormalizingFlow

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for the ConditionalNormalizingFlow model.

    Handles optimization with AdamW and a cosine LR schedule with linear
    warmup, gradient clipping, best-model checkpointing, and early stopping.

    Parameters
    ----------
    model : ConditionalNormalizingFlow
        The model to train (TFT + MAF).
    train_loader : DataLoader
        Training data batches of (macro_seq, returns).
    val_loader : DataLoader
        Validation data batches of (macro_seq, returns).
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularization coefficient.
    n_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience in epochs without improvement.
    grad_clip : float
        Maximum gradient norm for clipping.
    checkpoint_path : str
        File path to save the best model checkpoint.
    device : torch.device, optional
        Device to train on. Defaults to CUDA if available, else CPU.
    warmup_epochs : int
        Number of epochs for linear learning rate warmup.
    """

    def __init__(
        self,
        model: ConditionalNormalizingFlow,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: Optional[torch.device] = None,
        warmup_epochs: int = 2,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path
        self.warmup_epochs = warmup_epochs

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)

        logger.info("Training on device: %s", self.device)
        params = self.model.count_parameters()
        logger.info(
            "Model parameters: TFT=%d, Flow=%d, Total=%d",
            params["tft"], params["flow"], params["total"],
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-6,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(n_epochs - warmup_epochs, 1),
            eta_min=lr * 1e-2,
        )

        self.train_nll_history: List[float] = []
        self.val_nll_history: List[float] = []
        self.best_val_nll: float = float("inf")
        self.best_epoch: int = 0

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _set_warmup_lr(self, epoch: int) -> None:
        """Apply linear LR warmup for the first warmup_epochs epochs."""
        if epoch < self.warmup_epochs:
            lr = self.lr * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """
        Run one epoch and return mean NLL.

        Parameters
        ----------
        loader : DataLoader
            Data batches of (macro_seq, returns).
        train : bool
            If True, compute gradients and update parameters.

        Returns
        -------
        float
            Mean NLL over all batches in the epoch.
        """
        self.model.train(train)
        total_nll = 0.0
        n_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for macro_seq, returns in loader:
                macro_seq = macro_seq.to(self.device)
                returns = returns.to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                nll, _ = self.model(returns, macro_seq)

                if not torch.isfinite(nll):
                    logger.warning("Non-finite NLL encountered; skipping batch.")
                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue

                if train:
                    nll.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if not torch.isfinite(grad_norm):
                        logger.warning("Non-finite gradient norm; skipping optimizer step.")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.optimizer.step()

                total_nll += nll.item()
                n_batches += 1

        return total_nll / n_batches if n_batches > 0 else float("inf")

    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns
        -------
        dict
            Keys 'train_nll' and 'val_nll': per-epoch NLL histories.
        """
        logger.info("Starting training for up to %d epochs...", self.n_epochs)
        no_improve_count = 0

        for epoch in range(self.n_epochs):
            self._set_warmup_lr(epoch)

            train_nll = self._run_epoch(self.train_loader, train=True)
            val_nll = self._run_epoch(self.val_loader, train=False)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_nll_history.append(train_nll)
            self.val_nll_history.append(val_nll)

            logger.info(
                "Epoch %3d/%d | Train NLL: %.4f | Val NLL: %.4f | LR: %.2e",
                epoch + 1, self.n_epochs, train_nll, val_nll, current_lr,
            )

            if val_nll < self.best_val_nll:
                self.best_val_nll = val_nll
                self.best_epoch = epoch + 1
                no_improve_count = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_nll": val_nll,
                    "train_nll": train_nll,
                }, self.checkpoint_path)
                logger.info("  ✓ Best model saved (val NLL: %.4f)", val_nll)
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs). "
                    "Best epoch: %d (val NLL: %.4f)",
                    epoch + 1, self.patience, self.best_epoch, self.best_val_nll,
                )
                break

        logger.info(
            "Training complete. Best model at epoch %d with val NLL: %.4f",
            self.best_epoch, self.best_val_nll,
        )

        return {"train_nll": self.train_nll_history, "val_nll": self.val_nll_history}

    def load_best_model(self) -> None:
        """Load the best saved model checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded best model from epoch %d (val NLL: %.4f)",
            checkpoint["epoch"], checkpoint["val_nll"],
        )