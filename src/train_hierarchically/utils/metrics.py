"""Common evaluation metrics — thin wrappers for convenience."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(mean_squared_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def accuracy_sem(acc: float, n: int) -> float:
    """Standard error of accuracy (Wald / binomial proportion SE).

    Parameters
    ----------
    acc : float
        Classification accuracy in [0, 1].
    n : int
        Number of test samples.

    Returns
    -------
    float
        ``sqrt(acc * (1 - acc) / n)``
    """
    return float(np.sqrt(acc * (1.0 - acc) / n))


def covariance_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Covariance loss: L = -(1/n) Σ y_μ ŷ_μ.

    Unlike accuracy or MSE this requires *raw* model predictions
    (not argmax class indices).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth scalar targets.
    y_pred : np.ndarray
        Raw model predictions.

    Returns
    -------
    float
        The negated mean of the element-wise product.
    """
    return -float(np.mean(np.asarray(y_true) * np.asarray(y_pred)))


def ridge_test_error(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit RidgeCV on train data and evaluate on test data.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and targets.
    X_test, y_test : np.ndarray
        Test features and targets.
    alphas : np.ndarray
        Regularisation strengths to try.

    Returns
    -------
    tuple[float, float, float, float]
        ``(mse, std, sem, alpha)`` — test MSE, standard deviation of
        residuals, standard error of the mean, and chosen alpha.
    """
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    mse_val = float(np.mean(residuals**2))
    std_val = float(np.std(residuals))
    sem_val = std_val / (len(y_test) ** 0.5)
    return mse_val, std_val, sem_val, float(model.alpha_)
