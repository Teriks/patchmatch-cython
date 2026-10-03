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
