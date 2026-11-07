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

import numpy as np
import pytest

from patchmatch_cython import UnsupportedOption, inpaint_philippeau, inpaint_pyramid, inpaint_single


def _image_and_mask(size=40):
    y, x = np.mgrid[0:size, 0:size]
    image = np.stack(
        [
            (x * 3 + y) % 256,
            (y * 5) % 256,
            ((x + y) * 2) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[12:28, 14:30] = 255
    return image, mask


def test_nonsquare_odd_image_fills_the_hole():
    height, width = 31, 57
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack(
        [(x * 3) % 256, (y * 5) % 256, ((x + y) * 2) % 256],
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:22, 20:40] = 255
    result = inpaint_philippeau(image, mask, patch_size=2, seed=5)
    assert result.shape == (height, width, 3)
    known = mask == 0
    assert np.array_equal(result[known], image[known])
    assert not np.array_equal(result[mask != 0], image[mask != 0])


def test_known_pixels_are_unchanged_and_hole_is_filled():
    image, mask = _image_and_mask()
    result = inpaint_philippeau(image, mask, patch_size=2, seed=1212)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    known = mask == 0
    assert np.array_equal(result[known], image[known])
    assert not np.array_equal(result[mask != 0], image[mask != 0])


def test_same_seed_is_deterministic():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 8:28] = 255
    a = inpaint_philippeau(image, mask, patch_size=2, seed=7)
    b = inpaint_philippeau(image, mask, patch_size=2, seed=7)
    c = inpaint_philippeau(image, mask, patch_size=2, seed=99)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bool_mask_and_pyramid_option_match_direct_call():
    image, mask = _image_and_mask()
    direct = inpaint_philippeau(image, mask, patch_size=2, seed=3)
    via_flag = inpaint_pyramid(image, mask.astype(bool), patch_size=2, seed=3, algorithm="philippeau")
    assert np.array_equal(direct, via_flag)


def test_global_mask_pixels_stay_put():
    image, mask = _image_and_mask()
    global_mask = np.zeros(mask.shape, dtype=np.uint8)
    global_mask[0:8, 0:8] = 255
    result = inpaint_philippeau(image, mask, global_mask=global_mask, patch_size=2, seed=1)
    assert np.array_equal(result[0:8, 0:8], image[0:8, 0:8])


def test_white_pixels_are_the_default_holes():
    image, _ = _image_and_mask(24)
    image = image.copy()
    image[8:16, 8:16] = 255
    result = inpaint_philippeau(image, patch_size=2, seed=1)
    assert np.array_equal(result[:8], image[:8])
    assert not np.array_equal(result[8:16, 8:16], image[8:16, 8:16])


def test_philippeau_rejects_patchmatch_options():
    image, mask = _image_and_mask(24)
    with pytest.raises(UnsupportedOption, match="does not support em_loops") as em_loops_error:
        inpaint_pyramid(image, mask, algorithm="philippeau", em_loops=1, seed=1)
    assert em_loops_error.value.options == ("em_loops",)
    with pytest.raises(UnsupportedOption, match="does not support iterations, poisson_blend"):
        inpaint_single(
            image,
            mask,
            algorithm="philippeau",
            iterations=5,
            poisson_blend=True,
            seed=1,
        )
    with pytest.raises(TypeError, match="not_an_option"):
        inpaint_pyramid(image, mask, not_an_option=1)


def test_rejects_bad_patch_size_and_algorithm_name():
    image, mask = _image_and_mask(24)
    with pytest.raises(ValueError):
        inpaint_philippeau(image, mask, patch_size=0, seed=1)
    with pytest.raises(ValueError):
        inpaint_pyramid(image, mask, algorithm="other")


def test_omitted_seed_is_drawn():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    a = inpaint_philippeau(image, mask, patch_size=2)
    b = inpaint_philippeau(image, mask, patch_size=2)
    via_flag = inpaint_pyramid(image, mask, patch_size=2, algorithm="philippeau")
    assert a.shape == image.shape
    assert not np.array_equal(a, b)
    assert via_flag.shape == image.shape


def test_regularity_is_deterministic_and_keeps_global_mask():
    image, mask = _image_and_mask(36)
    global_mask = np.zeros(mask.shape, dtype=np.uint8)
    global_mask[:4, :] = 255
    ijmap = np.zeros(image.shape, dtype=np.float32)
    ijmap[..., 0] = np.linspace(0, 1, image.shape[1], dtype=np.float32)
    ijmap[..., 1] = np.linspace(0, 1, image.shape[0], dtype=np.float32)[:, None]
    kwargs = dict(
        global_mask=global_mask,
        ijmap=ijmap,
        patch_size=3,
        seed=4,
        guide_weight=0.4,
    )
    first = inpaint_philippeau(image, mask, **kwargs)
    second = inpaint_philippeau(image, mask, **kwargs)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:4], image[:4])
    known = (mask == 0) & (global_mask == 0)
    assert np.array_equal(first[known], image[known])
