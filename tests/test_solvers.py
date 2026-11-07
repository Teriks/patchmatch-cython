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

"""Solver-level behavior tests for Python and Cython implementations."""

from __future__ import annotations

import numpy as np
import pytest

from patchmatch_cython.python_solver import PythonSolver

from helpers import assert_hole_filled, requires_cython


def _make_case(size: int = 40, hole: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[:, : size // 2] = (50.0, 100.0, 150.0)
    image[:, size // 2 :] = (180.0, 60.0, 30.0)
    image += rng.normal(0.0, 2.0, size=image.shape).astype(np.float32)
    image = np.clip(image, 0, 255)
    mask = np.zeros((size, size), dtype=bool)
    c0 = size // 2 - hole // 2
    c1 = c0 + hole
    mask[c0:c1, c0:c1] = True
    working = image.copy()
    working[mask] = 0
    return working, mask


class TestPythonSolver:
    def test_fills_hole_with_finite_values(self):
        image, mask = _make_case()
        result = PythonSolver(image, mask, patch_size=5)(iterations=4, seed=0)
        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert_hole_filled(result, mask)

    def test_seed_reproducible(self):
        image, mask = _make_case()
        a = PythonSolver(image, mask, patch_size=5)(iterations=3, seed=7)
        b = PythonSolver(image, mask, patch_size=5)(iterations=3, seed=7)
        c = PythonSolver(image, mask, patch_size=5)(iterations=3, seed=8)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_nnf_shape_and_valid_targets(self):
        image, mask = _make_case()
        solver = PythonSolver(image, mask, patch_size=5)
        solver(iterations=2, seed=0)
        nnf = solver.nnf
        assert nnf.shape == (image.shape[0], image.shape[1], 2)
        assert nnf.dtype == np.int32
        query_y, query_x = np.where(solver._query_mask)
        for y, x in zip(query_y, query_x):
            mx, my = nnf[y, x]
            assert solver._is_valid_coords[my, mx]

    def test_requires_nnf_when_not_randomizing(self):
        image, mask = _make_case()
        solver = PythonSolver(image, mask, patch_size=5)
        with pytest.raises(ValueError, match="nnf must be provided"):
            solver(iterations=1, randomize_nnf=False, nnf=None)

    def test_preserves_known_pixels_in_working_image(self):
        image, mask = _make_case()
        known = image.copy()
        known[mask] = 0
        result = PythonSolver(known, mask, patch_size=5)(iterations=2, seed=0)
        assert np.allclose(result[~mask], known[~mask])


@requires_cython
class TestCythonSolver:
    def test_fills_hole_with_finite_values(self):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = _make_case()
        result = CythonSolver(image, mask, patch_size=5)(iterations=4, seed=0)
        assert result.dtype == np.float32
        assert_hole_filled(result, mask)

    def test_seed_reproducible(self):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = _make_case()
        a = CythonSolver(image, mask, patch_size=5)(iterations=3, seed=11)
        b = CythonSolver(image, mask, patch_size=5)(iterations=3, seed=11)
        c = CythonSolver(image, mask, patch_size=5)(iterations=3, seed=12)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_accepts_uint8_255_mask(self):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = _make_case()
        mask_u8 = mask.astype(np.uint8) * 255
        result = CythonSolver(image, mask_u8, patch_size=5)(iterations=2, seed=0)
        assert_hole_filled(result, mask)

    def test_nnf_points_to_valid_centers(self):
        from patchmatch_cython.cython_solver import CythonSolver
        from patchmatch_cython.imageops import (
            build_query_patch_centers,
            build_valid_patch_centers,
        )

        image, mask = _make_case()
        solver = CythonSolver(image, mask, patch_size=5)
        solver(iterations=2, seed=0)
        nnf = solver.nnf
        query = build_query_patch_centers(mask, half_patch_size=2)
        valid = build_valid_patch_centers(mask, half_patch_size=2)
        query_y, query_x = np.where(query)
        for y, x in zip(query_y, query_x):
            mx, my = nnf[y, x]
            assert valid[my, mx]


@requires_cython
class TestSolverParity:
    def test_similar_hole_statistics(self):
        """Python and Cython need not match bit-exact, but should fill similarly."""
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = _make_case(size=36, hole=8, seed=3)
        py = PythonSolver(image, mask, patch_size=5)(iterations=5, seed=0)
        cy = CythonSolver(image, mask, patch_size=5)(iterations=5, seed=0)

        assert_hole_filled(py, mask)
        assert_hole_filled(cy, mask)
        # Same known pixels untouched.
        assert np.allclose(py[~mask], image[~mask])
        assert np.allclose(cy[~mask], image[~mask])

        py_mean = float(np.mean(py[mask]))
        cy_mean = float(np.mean(cy[mask]))
        # Means should be in the same ballpark for this two-tone image.
        assert abs(py_mean - cy_mean) < 80.0
