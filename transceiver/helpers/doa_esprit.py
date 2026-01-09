#!/usr/bin/env python3
"""ESPRIT-based AoA estimation helpers for two-channel RX data."""

from __future__ import annotations

import numpy as np


def _validate_inputs(
    rx_data: np.ndarray, antenna_spacing: float, wavelength: float
) -> tuple[np.ndarray, float, float]:
    if antenna_spacing <= 0 or wavelength <= 0:
        raise ValueError("Antenna spacing and wavelength must be positive.")
    data = np.asarray(rx_data)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("rx_data must be a 2xN array.")
    if data.shape[1] < 2:
        raise ValueError("rx_data must contain at least two samples per channel.")
    return data[:2], antenna_spacing, wavelength


def estimate_aoa_esprit(
    rx_data: np.ndarray, antenna_spacing: float, wavelength: float
) -> float:
    """Estimate AoA (degrees) using a two-element ESPRIT formulation."""
    data, antenna_spacing, wavelength = _validate_inputs(
        rx_data, antenna_spacing, wavelength
    )
    data = data - np.mean(data, axis=1, keepdims=True)
    if np.allclose(data, 0):
        return float("nan")
    cov = data @ data.conj().T / data.shape[1]
    eigvals, eigvecs = np.linalg.eigh(cov)
    signal_vec = eigvecs[:, np.argmax(eigvals)]
    phase = np.angle(signal_vec[1] * np.conj(signal_vec[0]))
    sin_theta = phase * wavelength / (2 * np.pi * antenna_spacing)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_theta)))


def estimate_aoa_esprit_series(
    rx_data: np.ndarray,
    antenna_spacing: float,
    wavelength: float,
    snapshot_size: int = 1024,
    step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate AoA over sliding snapshots and return center indices + angles."""
    data, antenna_spacing, wavelength = _validate_inputs(
        rx_data, antenna_spacing, wavelength
    )
    if snapshot_size <= 0:
        raise ValueError("snapshot_size must be positive.")
    if step is None:
        step = max(1, snapshot_size // 2)
    if data.shape[1] < snapshot_size:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    centers = []
    angles = []
    for start in range(0, data.shape[1] - snapshot_size + 1, step):
        window = data[:, start : start + snapshot_size]
        angle = estimate_aoa_esprit(window, antenna_spacing, wavelength)
        if not np.isnan(angle):
            centers.append(start + snapshot_size / 2)
            angles.append(angle)
    return np.asarray(centers, dtype=np.float32), np.asarray(angles, dtype=np.float32)
