"""Shared utilities for wavelet scattering layers."""

from __future__ import annotations


def n_scattering_coefficients(j: int, n_angles: int, max_order: int) -> int:
    """Number of scattering path coefficients for given parameters.

    Parameters
    ----------
    j : int
        Number of scales (J parameter).
    n_angles : int
        Number of wavelet orientations (L parameter).
    max_order : int
        Maximum scattering path order (1 or 2).
    """
    n = 1  # zeroth order (low-pass)
    if max_order >= 1:
        n += j * n_angles
    if max_order >= 2:
        n += n_angles * n_angles * j * (j - 1) // 2
    return n


def scattering_output_spatial(h: int, w: int, j: int) -> tuple[int, int]:
    """Output spatial dimensions after wavelet scattering.

    Kymatio downsamples by ``2**J`` via ``H // 2**J``.
    """
    return h // (2**j), w // (2**j)
