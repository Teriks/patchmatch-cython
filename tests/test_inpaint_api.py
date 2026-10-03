"""Public API tests for inpaint_single / inpaint_pyramid."""

from __future__ import annotations

import numpy as np
import pytest

from patchmatch_cython.inpaint import inpaint_pyramid, inpaint_single
from patchmatch_cython.python_solver import PythonSolver

from helpers import assert_hole_filled, assert_known_pixels_preserved, requires_cython


@pytest.fixture
def api_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((96, 96, 3), dtype=np.float32)
    image[:, :48] = (35.0, 70.0, 105.0)
    image[:, 48:] = (210.0, 140.0, 40.0)
    image += rng.normal(0.0, 2.5, size=image.shape).astype(np.float32)
    image = np.clip(image, 0, 255)
    mask = np.zeros((96, 96), dtype=bool)
    mask[36:60, 36:60] = True
    return image, mask


class TestInpaintSingle:
    def test_output_shape_dtype(self, api_case):
        image, mask = api_case
        result = inpaint_single(image, mask, iterations=3, seed=0)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_preserves_known_pixels(self, api_case):
        image, mask = api_case
        result = inpaint_single(image, mask, iterations=3, seed=0)
        assert_known_pixels_preserved(image, result, mask)
        assert_hole_filled(result, mask)

    def test_accepts_bool_and_uint8_masks(self, api_case):
        image, mask = api_case
        a = inpaint_single(image, mask.astype(bool), iterations=2, seed=1)
        b = inpaint_single(image, mask.astype(np.uint8), iterations=2, seed=1)
        c = inpaint_single(image, (mask.astype(np.uint8) * 255), iterations=2, seed=1)
        assert a.shape == b.shape == c.shape
        assert_hole_filled(a, mask)
        assert_hole_filled(b, mask)
        assert_hole_filled(c, mask)

    def test_seed_reproducible(self, api_case):
        image, mask = api_case
        a = inpaint_single(image, mask, iterations=3, seed=42)
        b = inpaint_single(image, mask, iterations=3, seed=42)
        c = inpaint_single(image, mask, iterations=3, seed=43)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_empty_mask_returns_original(self, api_case):
        image, _ = api_case
        mask = np.zeros(image.shape[:2], dtype=bool)
        result = inpaint_single(image, mask, iterations=1, seed=0)
        expected = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        assert np.array_equal(result, expected)

    def test_grayscale_expands_channel(self):
        image = np.linspace(0, 255, 32 * 32, dtype=np.float32).reshape(32, 32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 12:20] = True
        result = inpaint_single(image, mask, iterations=2, seed=0)
        assert result.shape == (32, 32, 1)
        assert_hole_filled(result, mask)

    def test_explicit_python_solver(self, api_case):
        image, mask = api_case
        result = inpaint_single(
            image, mask, iterations=2, seed=0, solver_class=PythonSolver
        )
        assert result.dtype == np.uint8
        assert_known_pixels_preserved(image, result, mask)

    @requires_cython
    def test_explicit_cython_solver(self, api_case):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = api_case
        result = inpaint_single(
            image, mask, iterations=2, seed=0, solver_class=CythonSolver
        )
        assert result.dtype == np.uint8
        assert_known_pixels_preserved(image, result, mask)


class TestInpaintPyramid:
    def test_output_shape_dtype(self, api_case):
        image, mask = api_case
        result = inpaint_pyramid(image, mask, seed=0)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_preserves_known_and_fills_hole(self, api_case):
        image, mask = api_case
        result = inpaint_pyramid(image, mask, seed=0)
        assert_known_pixels_preserved(image, result, mask)
        assert_hole_filled(result, mask)

    def test_seed_reproducible(self, api_case):
        image, mask = api_case
        a = inpaint_pyramid(image, mask, seed=5)
        b = inpaint_pyramid(image, mask, seed=5)
        assert np.array_equal(a, b)

    def test_small_image_falls_back(self):
        image = np.random.default_rng(0).random((20, 20, 3), dtype=np.float32) * 255
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        result = inpaint_pyramid(image, mask, patch_size=5, seed=0)
        assert result.shape == image.shape
        assert_hole_filled(result, mask)

    def test_does_not_mutate_input(self, api_case):
        image, mask = api_case
        image_before = image.copy()
        mask_before = mask.copy()
        _ = inpaint_pyramid(image, mask, seed=0)
        assert np.array_equal(image, image_before)
        assert np.array_equal(mask, mask_before)

    @requires_cython
    def test_python_and_cython_both_succeed(self, api_case):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = api_case
        py = inpaint_pyramid(image, mask, seed=0, solver_class=PythonSolver)
        cy = inpaint_pyramid(image, mask, seed=0, solver_class=CythonSolver)
        assert_known_pixels_preserved(image, py, mask)
        assert_known_pixels_preserved(image, cy, mask)
        assert_hole_filled(py, mask)
        assert_hole_filled(cy, mask)
