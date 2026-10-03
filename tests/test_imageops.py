"""Unit tests for imageops helpers."""

import numpy as np
import pytest

from patchmatch_cython.imageops import (
    as_bool_mask,
    build_feature_image,
    build_source_ring_centers,
    build_query_patch_centers,
    build_spatial_vote_weights,
    build_valid_patch_centers,
    downsample,
    feather_compose,
    hole_distance_to_known,
    poisson_blend,
    prepare_inpaint_inputs,
    resize_image,
    sanitize_nnf,
    to_uint8_image,
    upsample,
)


class TestMaskAndCasting:
    def test_as_bool_mask_bool_passthrough(self):
        mask = np.array([[False, True], [True, False]])
        out = as_bool_mask(mask)
        assert out.dtype == np.bool_
        assert np.array_equal(out, mask)

    def test_as_bool_mask_uint8_ones(self):
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[1, 1] = 1
        assert as_bool_mask(mask).sum() == 1

    def test_as_bool_mask_uint8_255(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        out = as_bool_mask(mask)
        assert out.dtype == np.bool_
        assert out.sum() == 4

    def test_to_uint8_rounds_and_clips(self):
        image = np.array([[[-0.6, 10.6, 255.4], [255.6, 0.4, 128.5]]], dtype=np.float32)
        out = to_uint8_image(image)
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0
        assert out[0, 0, 1] == 11
        assert out[0, 0, 2] == 255
        assert out[0, 1, 0] == 255
        assert out[0, 1, 1] == 0
        assert out[0, 1, 2] == 128


class TestPatchCenterMaps:
    def test_valid_centers_exclude_hole_overlap(self):
        mask = np.zeros((21, 21), dtype=bool)
        mask[10, 10] = True
        valid = build_valid_patch_centers(mask, half_patch_size=2)
        assert not valid[10, 10]
        assert not valid[10, 12]
        assert valid[10, 13]

    def test_query_centers_include_border_band(self):
        mask = np.zeros((21, 21), dtype=bool)
        mask[10, 10] = True
        query = build_query_patch_centers(mask, half_patch_size=2)
        assert query[10, 10]
        assert query[10, 12]
        assert not query[10, 13]

    def test_valid_fallback_when_hole_dominates(self):
        mask = np.ones((9, 9), dtype=bool)
        mask[0, 0] = False  # only one known pixel
        valid = build_valid_patch_centers(mask, half_patch_size=2)
        # Full-patch-valid set is empty, so fallback keeps unmasked pixels.
        assert valid[0, 0]
        assert valid.sum() == 1

    def test_spatial_vote_weights_peak_at_center(self):
        weights = build_spatial_vote_weights(2)
        assert weights.shape == (5, 5)
        assert weights[2, 2] == pytest.approx(1.0)
        assert weights[2, 2] > weights[0, 0]


class TestSanitizeNnf:
    def test_rewrites_invalid_matches(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True
        valid = build_valid_patch_centers(mask, half_patch_size=1)

        nnf = np.zeros((16, 16, 2), dtype=np.int32)
        nnf[..., 0], nnf[..., 1] = np.meshgrid(np.arange(16), np.arange(16))
        nnf[7, 7] = (7, 7)

        cleaned = sanitize_nnf(nnf, mask, valid, rng=np.random.default_rng(0))
        mx, my = cleaned[7, 7]
        assert valid[my, mx]

    def test_clamps_out_of_bounds(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[3, 3] = True
        valid = build_valid_patch_centers(mask, half_patch_size=1)
        nnf = np.zeros((8, 8, 2), dtype=np.int32)
        nnf[3, 3] = (100, -5)
        cleaned = sanitize_nnf(nnf, mask, valid, rng=np.random.default_rng(1))
        mx, my = cleaned[3, 3]
        assert 0 <= mx < 8
        assert 0 <= my < 8
        assert valid[my, mx]


class TestResizeOps:
    def test_downsample_mean_pooling(self):
        from patchmatch_cython import imageops

        image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        # Force NumPy path for exact block-mean semantics.
        prev = imageops.HAS_OPENCV
        imageops.set_use_opencv(False)
        try:
            out = downsample(image, 2, 2)
            assert out.shape == (2, 2, 1)
            assert out[0, 0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)
        finally:
            imageops.set_use_opencv(prev)

    def test_downsample_opencv_matches_numpy_closely(self):
        from patchmatch_cython import imageops

        if not imageops.opencv_available():
            pytest.skip("OpenCV not installed")

        image = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
        prev = imageops.HAS_OPENCV
        try:
            imageops.set_use_opencv(False)
            numpy_out = downsample(image, 4, 4)
            imageops.set_use_opencv(True)
            opencv_out = downsample(image, 4, 4)
            assert opencv_out.shape == numpy_out.shape
            assert np.allclose(opencv_out, numpy_out, atol=1e-4)
        finally:
            imageops.set_use_opencv(prev)

    def test_morphology_matches_integral_window_maps(self):
        from patchmatch_cython import imageops

        if not imageops.opencv_available():
            pytest.skip("OpenCV not installed")

        mask = np.zeros((31, 31), dtype=bool)
        mask[10:16, 12:18] = True

        prev = imageops.HAS_OPENCV
        try:
            imageops.set_use_opencv(False)
            valid_np = build_valid_patch_centers(mask, 2)
            query_np = build_query_patch_centers(mask, 2)
            imageops.set_use_opencv(True)
            valid_cv = build_valid_patch_centers(mask, 2)
            query_cv = build_query_patch_centers(mask, 2)
            assert np.array_equal(valid_np, valid_cv)
            assert np.array_equal(query_np, query_cv)
        finally:
            imageops.set_use_opencv(prev)

    def test_resize_image_identity(self):
        image = np.random.default_rng(0).random((10, 12, 3), dtype=np.float32)
        out = resize_image(image, 10, 12)
        assert out.shape == image.shape
        assert np.allclose(out, image)

    def test_resize_image_upsamples(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[..., 0] = 10.0
        image[..., 1] = 20.0
        image[..., 2] = 30.0
        out = resize_image(image, 8, 8)
        assert out.shape == (8, 8, 3)
        assert np.allclose(out[..., 0], 10.0, atol=1e-3)
        assert np.allclose(out[..., 1], 20.0, atol=1e-3)
        assert np.allclose(out[..., 2], 30.0, atol=1e-3)

    def test_upsample_nnf_scales_and_clips(self):
        nnf = np.zeros((4, 4, 2), dtype=np.int32)
        nnf[..., 0], nnf[..., 1] = np.meshgrid(np.arange(4), np.arange(4))
        out = upsample(nnf, 8, 8)
        assert out.shape == (8, 8, 2)
        assert out[..., 0].min() >= 0
        assert out[..., 0].max() <= 7
        assert out[..., 1].min() >= 0
        assert out[..., 1].max() <= 7


class TestFeaturesAndPoisson:
    def test_feature_image_rgb_has_gradient_channels(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[:, 4:] = 255.0
        feats = build_feature_image(image)
        assert feats.shape == (8, 8, 5)
        assert np.max(np.abs(feats[..., 3])) > 0

    def test_feature_image_grayscale(self):
        image = np.linspace(0, 255, 64, dtype=np.float32).reshape(8, 8)
        feats = build_feature_image(image)
        assert feats.shape == (8, 8, 3)  # gray + gx + gy

    def test_feature_image_edge_weight_scales_gradients(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[:, 4:] = 255.0
        base = build_feature_image(image, edge_weight=1.0)
        boosted = build_feature_image(image, edge_weight=3.0)
        assert np.allclose(boosted[..., :3], base[..., :3])
        assert np.allclose(boosted[..., 3:], base[..., 3:] * 3.0)

    def test_feature_image_edge_weight_zero_is_color_only(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[:, 4:] = 255.0
        feats = build_feature_image(image, edge_weight=0.0)
        assert feats.shape == (8, 8, 3)

    def test_feature_image_mask_zeros_hole_touching_gradients(self):
        image = np.ones((16, 16, 3), dtype=np.float32) * 120
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True
        image[mask] = 0  # zeroed hole
        feats = build_feature_image(image, edge_weight=1.0, mask=mask)
        # Known pixels next to the hole must not carry fake gradients.
        dil = mask.copy()
        dil[1:] |= mask[:-1]
        dil[:-1] |= mask[1:]
        dil[:, 1:] |= mask[:, :-1]
        dil[:, :-1] |= mask[:, 1:]
        ring = dil & ~mask
        assert float(np.max(np.abs(feats[ring][..., 3:]))) == 0.0

    def test_source_ring_is_valid_and_near_hole(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 12:20] = True
        ring = build_source_ring_centers(mask, half_patch_size=2)
        assert ring.dtype == bool
        assert np.any(ring)
        assert not np.any(ring & mask)

    def test_poisson_preserves_known_and_changes_composited_hole(self):
        source = np.ones((16, 16, 3), dtype=np.float32) * 50
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True
        guide = source.copy()
        guide[mask] = 200  # composited path: known matches source
        out = poisson_blend(source, mask, guide, iterations=40, blend_band=4.0)
        assert np.allclose(out[~mask], source[~mask])
        # Must not be a no-op on the hole rim / interior.
        assert not np.allclose(out[mask], guide[mask])
        assert np.mean(out[mask]) > 50

    def test_poisson_noop_on_empty_mask(self):
        source = np.ones((8, 8, 3), dtype=np.float32) * 12
        guide = np.ones((8, 8, 3), dtype=np.float32) * 99
        mask = np.zeros((8, 8), dtype=bool)
        out = poisson_blend(source, mask, guide)
        assert np.allclose(out, source)

    def test_hole_distance_zero_on_known(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        dist = hole_distance_to_known(mask)
        assert np.all(dist[~mask] == 0)
        assert np.all(dist[mask] > 0)

    def test_feather_compose_preserves_known(self):
        source = np.ones((16, 16, 3), dtype=np.float32) * 40
        filled = np.ones((16, 16, 3), dtype=np.float32) * 200
        mask = np.zeros((16, 16), dtype=bool)
        mask[5:11, 5:11] = True
        out = feather_compose(source, mask, filled, feather=3.0)
        assert np.allclose(out[~mask], source[~mask])
        assert not np.allclose(out[mask], filled[mask])


class TestPrepareInputs:
    def test_zeros_hole_and_keeps_original(self):
        image = np.ones((10, 10, 3), dtype=np.float32) * 77
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True
        original, working, mask_bool = prepare_inpaint_inputs(image, mask, 3)
        assert np.allclose(original[mask_bool], 77)
        assert np.allclose(working[mask_bool], 0)
        assert np.allclose(working[~mask_bool], 77)

    def test_expands_grayscale(self):
        image = np.ones((12, 12), dtype=np.float32) * 5
        mask = np.zeros((12, 12), dtype=bool)
        mask[5, 5] = True
        original, working, _ = prepare_inpaint_inputs(image, mask, 3)
        assert original.shape == (12, 12, 1)
        assert working.shape == (12, 12, 1)
