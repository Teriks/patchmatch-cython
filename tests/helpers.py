"""Shared helpers for patchmatch_cython tests."""

from __future__ import annotations

import numpy as np
import pytest


def cython_available() -> bool:
    try:
        from patchmatch_cython.cython_solver import CythonSolver  # noqa: F401
        from patchmatch_cython import cython_solver_impl  # noqa: F401
        return True
    except ImportError:
        return False


requires_cython = pytest.mark.skipif(
    not cython_available(),
    reason="Cython acceleration not available",
)


def assert_known_pixels_preserved(
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        atol: float = 0.51,
) -> None:
    """Known pixels should match the rounded original image."""
    mask_bool = np.asarray(mask) != 0
    expected = np.clip(np.rint(original), 0, 255).astype(np.uint8)
    if expected.ndim == 2:
        expected = expected[:, :, None]
    assert np.allclose(result[~mask_bool], expected[~mask_bool], atol=atol)


def assert_hole_filled(result: np.ndarray, mask: np.ndarray) -> None:
    mask_bool = np.asarray(mask) != 0
    hole = result[mask_bool]
    assert hole.size > 0
    assert np.all(np.isfinite(hole))
    assert float(np.mean(hole)) > 1.0
