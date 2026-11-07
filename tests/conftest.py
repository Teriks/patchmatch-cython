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

"""Pytest fixtures for patchmatch_cython tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `import helpers` when pytest loads tests as top-level modules.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def rgb_image(rng: np.random.Generator) -> np.ndarray:
    """Structured RGB test image with two constant regions."""
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[:, :32] = (40.0, 80.0, 120.0)
    image[:, 32:] = (200.0, 160.0, 20.0)
    noise = rng.normal(0.0, 3.0, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255)


@pytest.fixture
def center_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    return mask


@pytest.fixture
def small_hole_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((48, 48, 3), dtype=np.float32)
    image[:24] = 30.0
    image[24:] = 220.0
    image += rng.normal(0.0, 2.0, size=image.shape).astype(np.float32)
    image = np.clip(image, 0, 255)
    mask = np.zeros((48, 48), dtype=bool)
    mask[18:30, 18:30] = True
    return image, mask
