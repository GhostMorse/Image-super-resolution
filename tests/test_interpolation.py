"""Tests for the 1-D interpolation kernels (the notebook's graded checks)."""

import numpy as np
from scipy.interpolate import interp1d

from interpolation.kernels import cubic_kernel, interpolate, linear_kernel, nearest_neighbor_kernel

EPS = 1e-9
X = np.linspace(-2, 2, num=5)
Y = np.abs(X)            # [2, 1, 0, 1, 2]
SCALE = 2
X_NEW = np.linspace(-2, 2, num=SCALE * X.shape[0])


def test_nearest_matches_scipy():
    ref = interp1d(X, Y, kind="nearest", fill_value="extrapolate")(X_NEW)
    assert np.linalg.norm(interpolate(Y, SCALE, nearest_neighbor_kernel) - ref) < EPS


def test_linear_matches_scipy():
    ref = interp1d(X, Y, kind="linear", fill_value="extrapolate")(X_NEW)
    assert np.linalg.norm(interpolate(Y, SCALE, linear_kernel) - ref) < EPS


def test_cubic_matches_reference():
    ref = np.array([2.0, 1.648267009, 1.121418827, 0.5925925925, 0.0877914952,
                    0.0877914952, 0.5925925925, 1.121418827, 1.648267009, 2.0])
    assert np.linalg.norm(interpolate(Y, SCALE, cubic_kernel) - ref) < EPS


def test_output_length_scales():
    sig = np.sin(np.linspace(0, 2 * np.pi, 8))
    assert len(interpolate(sig, 3, linear_kernel)) == 24


def test_kernels_preserve_nodes_for_nearest():
    # Nearest-neighbour at integer positions returns the original samples.
    sig = np.array([5.0, 1.0, 9.0, 3.0])
    out = interpolate(sig, 1, nearest_neighbor_kernel)
    assert np.allclose(out, sig)
