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

"""Input validation and error-path tests."""

import numpy as np
import pytest

from patchmatch_cython.imageops import prepare_inpaint_inputs
from patchmatch_cython.inpaint import inpaint_pyramid, inpaint_single
from patchmatch_cython.python_solver import PythonSolver


def test_reject_even_patch_size_prepare():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:15, 10:15] = True
    with pytest.raises(ValueError, match="odd"):
        prepare_inpaint_inputs(image, mask, patch_size=4)


def test_reject_even_patch_size_solver():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:15, 10:15] = True
    with pytest.raises(ValueError, match="odd"):
        PythonSolver(image, mask, patch_size=6)


def test_reject_zero_patch_size():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4, 4] = True
    with pytest.raises(ValueError, match="odd"):
        prepare_inpaint_inputs(image, mask, patch_size=0)


def test_reject_full_mask():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="entire image"):
        prepare_inpaint_inputs(image, mask, patch_size=3)


def test_reject_mask_shape_mismatch():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 15), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        prepare_inpaint_inputs(image, mask, patch_size=3)


def test_reject_image_smaller_than_patch():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    with pytest.raises(ValueError, match="smaller than patch_size"):
        prepare_inpaint_inputs(image, mask, patch_size=5)


def test_reject_invalid_image_ndim():
    image = np.zeros((16,), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        prepare_inpaint_inputs(image, mask, patch_size=3)


def test_inpaint_single_propagates_validation_errors():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="entire image"):
        inpaint_single(image, mask, patch_size=5)


def test_inpaint_pyramid_propagates_validation_errors():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    mask = np.zeros((64, 63), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        inpaint_pyramid(image, mask, patch_size=5)
