"""Tests for em_loops / edge_weighted / reverse_search toggles."""

from __future__ import annotations

import numpy as np
import pytest

from patchmatch_cython.inpaint import inpaint_pyramid, inpaint_single
from patchmatch_cython.python_solver import PythonSolver

from helpers import assert_hole_filled, assert_known_pixels_preserved, requires_cython


@pytest.fixture
def case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[:, :32] = (40.0, 80.0, 120.0)
    image[:, 32:] = (200.0, 150.0, 50.0)
    # Horizontal edge through the hole region encourages structure matching.
    image[28:36, :] = (10.0, 220.0, 10.0)
    image += rng.normal(0.0, 1.5, size=image.shape).astype(np.float32)
    image = np.clip(image, 0, 255)
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    return image, mask


class TestPythonQualityOptions:
    def test_em_loops_runs_and_fills(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        result = PythonSolver(working, mask, patch_size=5)(
            iterations=2, seed=0, em_loops=2
        )
        assert_hole_filled(result, mask)
        assert np.allclose(result[~mask], working[~mask])

    def test_edge_weighted_runs_and_fills(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        result = PythonSolver(working, mask, patch_size=5, edge_weighted=True)(
            iterations=3, seed=0
        )
        assert_hole_filled(result, mask)

    def test_reverse_search_runs_and_fills(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        result = PythonSolver(working, mask, patch_size=5)(
            iterations=3, seed=0, reverse_search=True
        )
        assert_hole_filled(result, mask)

    def test_all_options_together(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        result = PythonSolver(working, mask, patch_size=5, edge_weighted=True)(
            iterations=2, seed=1, em_loops=2, reverse_search=True
        )
        assert_hole_filled(result, mask)

    def test_em_loops_must_be_positive(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        solver = PythonSolver(working, mask, patch_size=5)
        with pytest.raises(ValueError, match="em_loops"):
            solver(iterations=1, seed=0, em_loops=0)

    def test_options_are_seed_reproducible(self, case):
        image, mask = case
        working = image.copy()
        working[mask] = 0
        kwargs = dict(
            iterations=2,
            em_loops=2,
            reverse_search=True,
        )
        a = PythonSolver(working, mask, patch_size=5, edge_weighted=True)(
            seed=9, **kwargs
        )
        b = PythonSolver(working, mask, patch_size=5, edge_weighted=True)(
            seed=9, **kwargs
        )
        assert np.array_equal(a, b)


class TestInpaintApiQualityOptions:
    def test_inpaint_single_all_toggles(self, case):
        image, mask = case
        result = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            em_loops=2,
            edge_weighted=True,
            reverse_search=True,
            poisson_blend=True,
        )
        assert result.dtype == np.uint8
        assert_known_pixels_preserved(image, result, mask)
        assert_hole_filled(result, mask)

    def test_inpaint_pyramid_all_toggles(self, case):
        image, mask = case
        result = inpaint_pyramid(
            image,
            mask,
            seed=0,
            solver_class=PythonSolver,
            em_loops=2,
            edge_weighted=True,
            reverse_search=True,
            poisson_blend=False,
        )
        assert result.dtype == np.uint8
        assert_known_pixels_preserved(image, result, mask)
        assert_hole_filled(result, mask)

    def test_poisson_blend_toggle_changes_hole(self, case):
        image, mask = case
        with_blend = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=True,
            mask_feather=0.0,
        )
        without_blend = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            mask_feather=0.0,
        )
        assert_known_pixels_preserved(image, with_blend, mask)
        assert_known_pixels_preserved(image, without_blend, mask)
        assert_hole_filled(with_blend, mask)
        assert_hole_filled(without_blend, mask)
        assert not np.array_equal(with_blend[mask], without_blend[mask])

    def test_mask_feather_softens_and_preserves_known(self, case):
        image, mask = case
        hard = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            mask_feather=0.0,
        )
        soft = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            mask_feather=4.0,
        )
        assert_known_pixels_preserved(image, soft, mask)
        assert_hole_filled(soft, mask)
        assert not np.array_equal(hard[mask], soft[mask])

    def test_hole_refine_runs(self, case):
        image, mask = case
        base = inpaint_single(
            image, mask, iterations=2, seed=0, solver_class=PythonSolver, hole_refine=0
        )
        refined = inpaint_single(
            image, mask, iterations=2, seed=0, solver_class=PythonSolver, hole_refine=2
        )
        assert_known_pixels_preserved(image, refined, mask)
        assert_hole_filled(refined, mask)
        assert not np.array_equal(base, refined)

    def test_defaults_match_explicit_defaults(self, case):
        image, mask = case
        legacy = inpaint_single(
            image, mask, iterations=2, seed=0, solver_class=PythonSolver
        )
        explicit = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            em_loops=1,
            edge_weighted=False,
            reverse_search=False,
            poisson_blend=True,
            hole_refine=0,
            mask_feather=0.0,
        )
        assert np.array_equal(legacy, explicit)

    def test_invalid_options_rejected_at_api(self, case):
        image, mask = case
        with pytest.raises(ValueError, match="em_loops"):
            inpaint_single(image, mask, em_loops=0, solver_class=PythonSolver)
        with pytest.raises(ValueError, match="hole_refine"):
            inpaint_single(image, mask, hole_refine=-1, solver_class=PythonSolver)
        with pytest.raises(ValueError, match="mask_feather"):
            inpaint_pyramid(image, mask, mask_feather=-0.5, solver_class=PythonSolver)

    def test_toggles_do_not_mutate_user_mask(self, case):
        image, mask = case
        mask_before = mask.copy()
        _ = inpaint_pyramid(
            image,
            mask,
            seed=0,
            solver_class=PythonSolver,
            em_loops=2,
            edge_weighted=True,
            reverse_search=True,
            poisson_blend=False,
            hole_refine=1,
            mask_feather=2.0,
        )
        assert np.array_equal(mask, mask_before)


@requires_cython
class TestCythonQualityOptions:
    def test_cython_all_toggles(self, case):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = case
        working = image.copy()
        working[mask] = 0
        result = CythonSolver(working, mask, patch_size=5, edge_weighted=True)(
            iterations=2, seed=0, em_loops=2, reverse_search=True
        )
        assert result.dtype == np.float32
        assert_hole_filled(result, mask)

    def test_inpaint_pyramid_cython_toggles(self, case):
        from patchmatch_cython.cython_solver import CythonSolver

        image, mask = case
        result = inpaint_pyramid(
            image,
            mask,
            seed=0,
            solver_class=CythonSolver,
            em_loops=2,
            edge_weighted=True,
            reverse_search=True,
        )
        assert_known_pixels_preserved(image, result, mask)
        assert_hole_filled(result, mask)
