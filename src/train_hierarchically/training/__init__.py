"""train_hierarchically.training — Model training orchestration.

Provides a :class:`Trainer` class that handles the training loop,
validation, callbacks, and logging in a customisable way.  Also provides
:class:`BackpropTrainer` and :class:`RidgeSpectralTrainer` for
gradient-based and ridge-spectral training respectively.
"""

from .backprop import BackpropConfig, BackpropTrainer
from .base import BaseTrainer, BaseTrainerConfig, fit_ridge_readout
from .cnn_ridge_spectral import CNNRidgeSpectralTrainer
from .ridge_spectral import RidgeSpectralConfig, RidgeSpectralTrainer

__all__ = [
    "BaseTrainer",
    "BaseTrainerConfig",
    "fit_ridge_readout",
    "BackpropConfig",
    "BackpropTrainer",
    "CNNRidgeSpectralTrainer",
    "RidgeSpectralConfig",
    "RidgeSpectralTrainer",
]
