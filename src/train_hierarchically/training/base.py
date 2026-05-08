"""Base trainer interface and shared configuration."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..utils.helpers import resolve_device

log = logging.getLogger(__name__)


@dataclass
class BaseTrainerConfig:
    device: str = "cpu"
    verbose: bool = True


def fit_ridge_readout(
    model: nn.Module,
    h_train: Tensor,
    y_train: Tensor,
    *,
    h_test: Tensor | None = None,
    y_test_np: np.ndarray | None = None,
    alpha_min: float = -6.0,
    alpha_max: float = 6.0,
    alpha_num: int = 500,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit a RidgeCV readout and store weights in *model*.

    Fits ``RidgeCV`` on the training features, stores
    ``ridge_weight`` / ``ridge_bias`` as frozen parameters on *model*,
    and computes train (and optionally test) MSE + binary accuracy.

    Parameters
    ----------
    model : nn.Module
        Model to attach ``ridge_weight`` and ``ridge_bias`` to.
    h_train : Tensor
        Training features ``(N_train, D)``.
    y_train : Tensor
        Training targets ``(N_train,)``.
    h_test : Tensor | None
        Test features ``(N_test, D)``.
    y_test_np : ndarray | None
        Test targets as NumPy array ``(N_test,)``.
    alpha_min, alpha_max, alpha_num : float
        Log10 bounds and count for regularization candidates.
    verbose : bool
        Whether to log ridge metrics.

    Returns
    -------
    dict[str, Any]
        Metrics dict with keys ``train_mse``, ``ridge_alpha``,
        ``accuracy``, and optionally ``test_mse``, ``test_sem``,
        ``test_accuracy``, ``test_accuracy_sem``.
    """
    alphas = np.logspace(alpha_min, alpha_max, alpha_num)

    h_train_np = h_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    ridge = RidgeCV(alphas=alphas)
    ridge.fit(h_train_np, y_train_np)

    # Store ridge weights in model
    model.ridge_weight = nn.Parameter(
        torch.tensor(
            ridge.coef_, dtype=h_train.dtype, device=h_train.device,
        ),
        requires_grad=False,
    )
    model.ridge_bias = nn.Parameter(
        torch.tensor(
            [ridge.intercept_],
            dtype=h_train.dtype,
            device=h_train.device,
        ),
        requires_grad=False,
    )

    # Training metrics
    y_pred_train = ridge.predict(h_train_np)
    train_mse = float(np.mean((y_train_np - y_pred_train) ** 2))
    ridge_alpha = float(ridge.alpha_)

    # Binary classification: threshold at 0 -> {-1, +1}
    y_pred_labels = np.where(y_pred_train >= 0, 1.0, -1.0)
    train_acc = float(np.mean(y_train_np == y_pred_labels))

    if verbose:
        log.info(
            "Ridge: alpha=%.4e, train_mse=%.6f",
            ridge_alpha, train_mse,
        )

    final: dict[str, Any] = {
        "train_mse": train_mse,
        "ridge_alpha": ridge_alpha,
        "accuracy": train_acc,
    }

    # Test metrics
    if h_test is not None and y_test_np is not None:
        h_test_np = h_test.cpu().numpy()
        y_pred_test = ridge.predict(h_test_np)
        test_residuals = (y_test_np - y_pred_test) ** 2
        test_mse = float(np.mean(test_residuals))
        test_sem = float(
            np.std(test_residuals) / (len(y_test_np) ** 0.5)
        )

        y_pred_test_labels = np.where(y_pred_test >= 0, 1.0, -1.0)
        test_acc = float(np.mean(y_test_np == y_pred_test_labels))

        test_acc_sem = float(
            np.sqrt(test_acc * (1.0 - test_acc) / len(y_test_np))
        )

        final["test_mse"] = test_mse
        final["test_sem"] = test_sem
        final["test_accuracy"] = test_acc
        final["test_accuracy_sem"] = test_acc_sem

        if verbose:
            log.info(
                "  Test: mse=%.6f, sem=%.6f, accuracy=%.4f±%.4f",
                test_mse, test_sem, test_acc, test_acc_sem,
            )

    return final


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : BaseTrainerConfig
        Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
    ) -> None:
        self.config = config
        self.config.device = resolve_device(config.device)
        self.model = model.to(self.config.device)

    @abstractmethod
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Train the model on *loader* and return it with results.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        **kwargs
            Subclass-specific keyword arguments.

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model and a results dict.
        """

    def _to_device(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """Move an arbitrary number of tensors to ``self.config.device``."""
        return tuple(t.to(self.config.device) for t in tensors)
