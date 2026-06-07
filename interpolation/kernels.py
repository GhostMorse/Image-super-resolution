"""1-D interpolation from scratch: a generic kernel-based resampler.

A signal is resampled to a new length by, for every output position, taking a
kernel-weighted average of the input samples (eq. ``g(x) = sum_k c_k u((x-x_k)/h)``).
Three symmetric kernels are provided: nearest-neighbour, linear, and the
Catmull-Rom cubic. Everything is fully vectorised (no Python loops).
"""

from __future__ import annotations

import numpy as np


def interpolate(signal: np.ndarray, scale_factor: float, kernel) -> np.ndarray:
    """Resample a 1-D signal to ``len(signal) * scale_factor`` points.

    :param signal: 1-D input signal.
    :param scale_factor: Output-to-input length ratio.
    :param kernel: Callable mapping a distance array to weights, both of shape
        ``(new_length, original_length)``.
    :returns: The interpolated 1-D signal.
    """
    original_length = len(signal)
    new_length = int(original_length * scale_factor)

    new_indices = np.linspace(0, original_length - 1, new_length)
    original_x = np.arange(original_length)

    distances = np.abs(new_indices[:, np.newaxis] - original_x[np.newaxis, :])
    weights = kernel(distances)

    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1
    weights_normalized = weights / weights_sum

    return np.dot(weights_normalized, signal)


def nearest_neighbor_kernel(distance: np.ndarray) -> np.ndarray:
    """Weight 1 for the single closest input sample, 0 elsewhere."""
    new_length, original_length = distance.shape
    weights = np.zeros((new_length, original_length))
    closest = np.argmin(distance, axis=1)
    weights[np.arange(new_length), closest] = 1
    return weights


def linear_kernel(distance: np.ndarray) -> np.ndarray:
    """Triangular kernel ``max(0, 1 - |d|)`` (linear interpolation)."""
    weights = np.zeros_like(distance, dtype=float)
    mask = distance <= 1
    weights[mask] = 1 - np.abs(distance[mask])
    return weights


def cubic_kernel(distance: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic convolution kernel (``a = -0.5``)."""
    weights = np.zeros_like(distance, dtype=float)
    m1 = (distance >= 0) & (distance < 1)
    m2 = (distance >= 1) & (distance < 2)
    weights[m1] = 1.5 * distance[m1] ** 3 - 2.5 * distance[m1] ** 2 + 1
    weights[m2] = -0.5 * distance[m2] ** 3 + 2.5 * distance[m2] ** 2 - 4 * distance[m2] + 2
    return weights
