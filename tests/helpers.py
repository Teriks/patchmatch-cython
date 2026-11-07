# BSD 3-Clause License
#
# Copyright (c) 2025, Teriks
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
