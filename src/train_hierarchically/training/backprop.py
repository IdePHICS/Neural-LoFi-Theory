"""Standard back-propagation trainer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from train_hierarchically.utils.metrics import accuracy

from .base import BaseTrainer, BaseTrainerConfig

log = logging.getLogger(__name__)


@dataclass
class BackpropConfig(BaseTrainerConfig):
    """Configuration for :class:`BackpropTrainer`.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate passed to the optimizer.
    optimizer_cls : type
        Optimizer class (must accept ``params`` and ``lr`` arguments).
    """

    epochs: int = 10
    lr: float = 1e-3
    optimizer_cls: type = torch.optim.Adam


class BackpropTrainer(BaseTrainer):
    """Trainer that uses standard gradient-based back-propagation.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : BackpropConfig
        Training hyper-parameters.
    loss_fn : Callable[[Tensor, Tensor], Tensor]
        Loss function accepting ``(prediction, target)`` and returning a
        scalar tensor.
    pred_fn : Callable[[Tensor], Tensor] | None
        Function to convert raw model output to predicted labels.
        Defaults to ``lambda p: p.argmax(dim=-1)`` (multi-class).
        For binary MSE with ±1 labels use
        ``lambda p: torch.sign(p.squeeze(-1))``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BackpropConfig,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        pred_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(model, config)
        self.loss_fn = loss_fn
        self.pred_fn: Callable[[Tensor], Tensor] = (
            pred_fn if pred_fn is not None else lambda p: p.argmax(dim=-1)
        )
        self.optimizer = config.optimizer_cls(model.parameters(), lr=config.lr)

    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Run a standard training loop for ``config.epochs`` epochs.

        Parameters
        ----------
        loader : DataLoader
            Training data loader yielding ``(X, y)`` batches.
        test_loader : DataLoader | None
            Optional test data loader for per-epoch evaluation.
        epoch_callbacks : list[Callable] | None
            Optional list of callbacks invoked at the end of each epoch.
            Each callback is called as ``cb(trainer, epoch, entry)`` where
            *trainer* is this trainer instance, *epoch* is the 0-based epoch
            index, and *entry* is the per-epoch metrics dict (mutable).

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model and a results dict with key ``"epochs"``
            containing a list of per-epoch metric dicts.
        """
        test_loader: DataLoader | None = kwargs.get("test_loader")  # type: ignore[assignment]
        epoch_callbacks: list[Callable[..., Any]] | None = kwargs.get("epoch_callbacks")  # type: ignore[assignment]
        epochs = self.config.epochs

        self.model.train()
        history: list[dict[str, Any]] = []

        for epoch in range(epochs):
            total_loss = 0.0
            all_y_true: list[np.ndarray] = []
            all_y_pred: list[np.ndarray] = []
            total_grad_norm = 0.0

            for x, y in loader:
                x, y = self._to_device(x, y)

                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss_fn(pred.squeeze(), y)
                loss.backward()

                total_grad_norm += self._global_grad_norm()
                self.optimizer.step()
                total_loss += loss.item()

                all_y_true.append(y.detach().cpu().numpy())
                all_y_pred.append(self.pred_fn(pred.detach()).cpu().numpy())

            n_batches = max(len(loader), 1)
            mean_loss = total_loss / n_batches
            lr = self.optimizer.param_groups[0]["lr"]
            train_acc = accuracy(np.concatenate(all_y_true), np.concatenate(all_y_pred))

            entry: dict[str, Any] = {
                "epoch": epoch,
                "loss": mean_loss,
                "accuracy": train_acc,
                "grad_norm": total_grad_norm / n_batches,
                "lr": lr,
            }

            # Evaluate on test set if available
            test_metrics = self._eval_on_test(test_loader)
            if test_metrics:
                entry.update({f"test_{k}": v for k, v in test_metrics.items()})

            # Invoke epoch callbacks
            if epoch_callbacks:
                for cb in epoch_callbacks:
                    cb(self, epoch, entry)

            history.append(entry)

            if self.config.verbose:
                msg = f"Epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}"
                if test_metrics:
                    parts = [f"{k}={v:.4f}" for k, v in test_metrics.items()]
                    msg += f"  test: [{', '.join(parts)}]"
                log.info(msg)

        return self.model, {"epochs": history}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_on_test(self, test_loader: DataLoader | None) -> dict[str, float] | None:
        """Evaluate the current model on *test_loader* if available."""
        if test_loader is None:
            return None
        self.model.eval()
        all_y: list[np.ndarray] = []
        all_p: list[np.ndarray] = []
        for x, y in test_loader:
            x, y = self._to_device(x, y)
            pred = self.model(x)
            all_y.append(y.cpu().numpy())
            all_p.append(self.pred_fn(pred).cpu().numpy())
        self.model.train()
        return {"accuracy": accuracy(np.concatenate(all_y), np.concatenate(all_p))}

    def _global_grad_norm(self) -> float:
        """Compute the global L2 gradient norm across all parameters."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total**0.5
