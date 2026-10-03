"""
Regression probes for weak / questionable compose and refine behaviors:

1. Is ``hole_refine`` a dead path, or just too weak after a finished solve?
2. Does ``mask_feather`` leak original hole pixels back into the rim?
"""

from __future__ import annotations

import numpy as np

import patchmatch_cython.inpaint as inpaint_mod
from patchmatch_cython.imageops import (
    feather_compose,
    hole_distance_to_known,
    poisson_blend,
)
from patchmatch_cython.inpaint import inpaint_single
from patchmatch_cython.python_solver import PythonSolver

from helpers import assert_hole_filled, assert_known_pixels_preserved


def _structured_case(rng: np.random.Generator, size: int = 48):
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[:, : size // 2] = (30.0, 60.0, 90.0)
    image[:, size // 2 :] = (180.0, 120.0, 40.0)
    image[size // 3 : 2 * size // 3, :] = (20.0, 200.0, 20.0)
    image += rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    image = np.clip(image, 0, 255)
    mask = np.zeros((size, size), dtype=bool)
    c0, c1 = size // 2 - 8, size // 2 + 8
    mask[c0:c1, c0:c1] = True
    return image, mask


class TestHoleRefineIsAlive:
    def test_hole_refine_pass_is_invoked(self, monkeypatch, rng):
        image, mask = _structured_case(rng)
        calls: list[int] = []
        real = inpaint_mod._hole_refine_pass

        def spy(*args, **kwargs):
            # positional: (original, mask, filled, nnf, patch_size, solver_class,
            #              hole_refine, ...)
            calls.append(int(args[6]))
            return real(*args, **kwargs)

        monkeypatch.setattr(inpaint_mod, "_hole_refine_pass", spy)
        _ = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            hole_refine=5,
        )
        assert calls == [5]

    def test_hole_refine_changes_under_iterated_solve(self, rng):
        """After a short main solve, refine iters must move the hole."""
        image, mask = _structured_case(rng)
        base = inpaint_single(
            image,
            mask,
            iterations=1,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            hole_refine=0,
        )
        refined = inpaint_single(
            image,
            mask,
            iterations=1,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            hole_refine=8,
        )
        assert_known_pixels_preserved(image, refined, mask)
        assert_hole_filled(refined, mask)
        diff = np.abs(
            base[mask].astype(np.int16) - refined[mask].astype(np.int16)
        ).astype(np.float32)
        assert float(diff.mean()) > 0.5
        assert not np.array_equal(base[mask], refined[mask])

    def test_hole_refine_diminishes_after_longer_main_solve(self, rng):
        """
        Same refine budget changes less after a longer main solve than after a
        short one — evidence of diminishing returns, not a dead code path.
        """
        image, mask = _structured_case(rng)

        def run(iterations: int, hole_refine: int):
            return inpaint_single(
                image,
                mask,
                iterations=iterations,
                seed=1,
                solver_class=PythonSolver,
                poisson_blend=False,
                hole_refine=hole_refine,
            )

        short_base = run(1, 0)
        short_ref = run(1, 6)
        long_base = run(6, 0)
        long_ref = run(6, 6)

        short_delta = float(
            np.abs(
                short_base[mask].astype(np.int16) - short_ref[mask].astype(np.int16)
            ).mean()
        )
        long_delta = float(
            np.abs(
                long_base[mask].astype(np.int16) - long_ref[mask].astype(np.int16)
            ).mean()
        )

        # Refine must do something on the short solve.
        assert short_delta > 0.25
        # After a longer solve, the same refine budget should move less (or equal).
        assert long_delta <= short_delta + 0.05

    def test_hole_refine_reuses_nnf_when_present(self, monkeypatch, rng):
        image, mask = _structured_case(rng)
        seen: list[bool] = []

        class TrackingSolver(PythonSolver):
            def __call__(self, *args, **kwargs):
                seen.append(bool(kwargs.get("randomize_nnf", args[1] if len(args) > 1 else True)))
                return super().__call__(*args, **kwargs)

        _ = inpaint_single(
            image,
            mask,
            iterations=2,
            seed=0,
            solver_class=TrackingSolver,
            poisson_blend=False,
            hole_refine=3,
        )
        # Main solve randomizes; refine pass must warm-start from NNF.
        assert seen[0] is True
        assert seen[-1] is False


class TestFeatherDoesNotLeakOriginalHole:
    def test_unit_feather_prefers_boundary_not_original_hole(self):
        """
        Known=green, original hole=red marker, filled=blue.

        RGB distance to red is a bad leak metric (green is closer to red than
        blue is). The red channel is unique to the original hole, so any rim
        rise in R is real leakage.
        """
        source = np.zeros((24, 24, 3), dtype=np.float32)
        source[:] = (0.0, 180.0, 0.0)  # known green
        mask = np.zeros((24, 24), dtype=bool)
        mask[8:16, 8:16] = True
        source[mask] = (255.0, 0.0, 0.0)  # distinctive original hole

        filled = source.copy()
        filled[mask] = (0.0, 0.0, 220.0)  # PatchMatch-like fill

        hard = source.copy()
        hard[mask] = filled[mask]
        soft = feather_compose(source, mask, filled, feather=4.0)

        dist = hole_distance_to_known(mask)
        rim = mask & (dist <= 2.5)
        assert np.any(rim)

        soft_rim = soft[rim].astype(np.float32)
        hard_rim = hard[rim].astype(np.float32)

        # Unique hole channel must not reappear in the feathered rim.
        assert float(np.mean(soft_rim[:, 0])) < 1.0
        assert float(np.mean(soft_rim[:, 0])) <= float(np.mean(hard_rim[:, 0])) + 0.5
        # Rim should pick up green from the harmonic pull and/or keep blue fill.
        assert float(np.mean(soft_rim[:, 1])) > float(np.mean(hard_rim[:, 1])) - 1.0
        assert np.allclose(soft[~mask], source[~mask])

    def test_poisson_blend_rim_not_pulled_to_original_hole(self):
        source = np.zeros((20, 20, 3), dtype=np.float32)
        source[:] = 40.0
        mask = np.zeros((20, 20), dtype=bool)
        mask[6:14, 6:14] = True
        source[mask] = 250.0  # original hole marker

        guide = source.copy()
        guide[mask] = 10.0

        out = poisson_blend(source, mask, guide, iterations=40, blend_band=4.0)
        dist = hole_distance_to_known(mask)
        rim = mask & (dist <= 2.0)
        # Rim should drop from the bright marker toward the dark fill / known gray.
        assert float(np.mean(out[rim])) < 100.0
        assert float(np.mean(out[rim])) < float(np.mean(source[rim]))

    def test_api_mask_feather_does_not_reintroduce_hole_marker(self, rng):
        image, mask = _structured_case(rng, size=40)
        # Paint a neon marker into the hole of the *original* only.
        marked = image.copy()
        marked[mask] = (255.0, 0.0, 255.0)

        hard = inpaint_single(
            marked,
            mask,
            iterations=3,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            mask_feather=0.0,
        )
        soft = inpaint_single(
            marked,
            mask,
            iterations=3,
            seed=0,
            solver_class=PythonSolver,
            poisson_blend=False,
            mask_feather=5.0,
        )

        dist = hole_distance_to_known(mask)
        rim = mask & (dist <= 2.5)
        marker = np.array([255.0, 0.0, 255.0], dtype=np.float32)
        hard_to_m = float(
            np.mean(np.linalg.norm(hard[rim].astype(np.float32) - marker, axis=1))
        )
        soft_to_m = float(
            np.mean(np.linalg.norm(soft[rim].astype(np.float32) - marker, axis=1))
        )
        assert_known_pixels_preserved(marked, soft, mask)
        # Soft rim must not move toward the neon original hole content.
        assert soft_to_m + 0.5 >= hard_to_m
