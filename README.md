# PatchMatch Cython

High-performance PatchMatch implementation for pyramidical image inpainting in Cython.

This package can function without OpenCV (`opencv-python` / `opencv-python-headless`) with only slightly reduced quality and speed in a few helpers. When OpenCV is installed it is used for resize (`INTER_LINEAR` / `INTER_AREA`), morphology for valid/query centers, and distance transforms for seam feathering.

The package provides a pure Python solver and a Cython solver; the Cython solver is typically about 30–50× faster depending on patch size.

## Installation

```bash
pip install patchmatch-cython
```

**From source:**
```bash
git clone <repository-url>
cd patchmatch-cython
pip install .
```

## Usage

### With OpenCV

```python
import cv2
from patchmatch_cython import inpaint_pyramid

image = cv2.imread('image.jpg')
mask = cv2.imread('mask.png', 0) > 128  # True = hole to fill

result = inpaint_pyramid(image, mask)
cv2.imwrite('result.jpg', result)
```

### With Pillow

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_pyramid

image = np.array(Image.open('image.jpg'))
mask = np.array(Image.open('mask.png').convert('L')) > 128

result = inpaint_pyramid(image, mask)  # auto-selects fastest available solver
Image.fromarray(result).save('result.jpg')
```

### Explicit solver selection

```python
from patchmatch_cython import PythonSolver, CythonSolver, inpaint_pyramid

result = inpaint_pyramid(image, mask, solver_class=CythonSolver)  # Fast
result = inpaint_pyramid(image, mask, solver_class=PythonSolver)  # Fallback
```

### Quality / compose options

Both `inpaint_pyramid` and `inpaint_single` accept the same optional knobs.
Defaults preserve a sensible baseline; turn options on when you need more structure,
completeness, or softer seams.

```python
from patchmatch_cython import inpaint_pyramid

result = inpaint_pyramid(
    image,
    mask,
    patch_size=5,          # odd int; square patch side
    seed=0,                # reproducible RNG (optional)
    em_loops=2,            # coarse match→vote→rematch loops (pyramid: coarse levels only)
    edge_weighted=True,    # structure-aware distance (boosted gradients)
    reverse_search=True,   # source→query completeness pass + stronger usage penalty
    poisson_blend=True,    # harmonic seam soften (default ~8px band)
    hole_refine=3,         # extra full-res PatchMatch iterations after the main solve
    mask_feather=6.0,      # soft compose width in px from the hole boundary
)
```

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | `5` | Odd positive int; square patch size for matching / voting. |
| `seed` | `None` | RNG seed for reproducible NNF init / search. |
| `em_loops` | `1` | Outer match→paint→rematch loops. On pyramids, applied at coarse levels only (finest stays `1`). |
| `edge_weighted` | `False` | `False` = color-only matching. `True` = color + boosted gradient channels. Hole-touching gradients are suppressed so zeroed holes do not create fake edges. |
| `reverse_search` | `False` | After each EM loop, try assigning underused near-hole sources to better queries (completeness). |
| `poisson_blend` | `True` | Soften seams by blending the PatchMatch fill with a harmonic membrane near the hole rim. Ignored when `mask_feather > 0` (feather supplies the band). |
| `hole_refine` | `0` | Extra full-resolution PatchMatch iterations after the main solve (`0` disables). Only hole pixels are painted; the user mask is not modified. |
| `mask_feather` | `0.0` | Soft compose width in pixels from the hole boundary. Uses a distance ramp toward a harmonic rim pull (not the original hole content). Does **not** change the mask used for matching. |

`inpaint_single` is the same API plus `iterations` (PatchMatch passes at one scale):

```python
from patchmatch_cython import inpaint_single

result = inpaint_single(
    image,
    mask,
    iterations=7,
    em_loops=2,
    reverse_search=True,
    hole_refine=2,
    mask_feather=4.0,
    poisson_blend=False,  # feather handles seams when mask_feather > 0
)
```

**Masks:** accept `bool`, `{0,1}`, or `{0,255}`. `True` / non-zero = region to inpaint. Known (unmasked) pixels are preserved in the output. The library never dilates/erodes the user mask for matching; `mask_feather` only affects final compositing.

**OpenCV runtime toggle** (optional):

```python
from patchmatch_cython.imageops import set_use_opencv, opencv_available

set_use_opencv(False)  # force NumPy fallbacks even if OpenCV is installed
```

## Package extras

**For examples:**

```bash
pip install patchmatch-cython[visualization]
```

Adds: `matplotlib`, `Pillow`

**For development:**

```bash
pip install patchmatch-cython[dev]
```

Adds: `pytest`, `build`, `cibuildwheel`, `setuptools`, `wheel`

## Building

To build wheels locally for distribution or development:

### Prerequisites

```bash
pip install cibuildwheel
```

### Build wheels

Use the included `local_build.py` script:

```bash
# Build for all supported Python versions (3.10-3.14)
python local_build.py

# Build for specific Python versions
python local_build.py --python 3.11 3.12

# Build for specific platform
python local_build.py --platform linux

# Build and test wheel installation
python local_build.py --test

# Build without cleaning previous artifacts
python local_build.py --no-clean

# View all options
python local_build.py --help
```

### Build options

- `--python`: Specify Python versions to build for (e.g., `3.11 3.12`)
- `--platform`: Target platform (`auto`, `linux`, `macos`, `windows`)
- `--test`: Test wheel installation after building
- `--no-clean`: Skip cleaning build artifacts before building

## Requirements

- Python 3.10+
- NumPy (auto-installed)
- C++ compiler (for building from source)
