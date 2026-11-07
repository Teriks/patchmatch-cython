# PatchMatch Cython

High-performance PatchMatch implementation for pyramidical image inpainting in Cython. The default solver is the Barnes-style pyramid below. `algorithm="philippeau"` runs the Xavier Philippeau pyramid, also in Cython.

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

Images are `HxWxC`. Masks are `HxW`: `True` or any non-zero value is the hole. The result is `uint8`, and pixels outside the hole are unchanged. `inpaint_pyramid` picks `CythonSolver` when the extension is built, and `PythonSolver` otherwise. Pass `solver_class=PythonSolver` to force the pure Python path.

### Remove a region from a photo

```python
import cv2
from patchmatch_cython import inpaint_pyramid

image = cv2.imread("photo.jpg")          # BGR uint8
mask = cv2.imread("hole.png", cv2.IMREAD_GRAYSCALE) > 128

result = inpaint_pyramid(image, mask, seed=42)
cv2.imwrite("filled.jpg", result)
```

The same job with Pillow. Convert to RGB so a PNG with an alpha channel stays three channels.

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_pyramid

image = np.array(Image.open("photo.jpg").convert("RGB"))
mask = np.array(Image.open("hole.png").convert("L")) > 128

result = inpaint_pyramid(image, mask, seed=42)
Image.fromarray(result).save("filled.jpg")
```

A rectangle is enough when you do not have a mask file. `seed` makes the fill repeatable; omit it and each run can differ.

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_pyramid

image = np.array(Image.open("photo.jpg").convert("RGB"))
mask = np.zeros(image.shape[:2], dtype=bool)
mask[120:200, 80:160] = True

result = inpaint_pyramid(image, mask, seed=42)
Image.fromarray(result).save("filled.jpg")
```

### Thin scratch, one scale

`inpaint_single` stays at full resolution. Use it for a narrow mark, where a pyramid has little to add. `iterations` is the number of match passes.

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_single

image = np.array(Image.open("photo.jpg").convert("RGB"))
mask = np.zeros(image.shape[:2], dtype=bool)
mask[240:244, 40:400] = True

result = inpaint_single(image, mask, iterations=5, seed=42)
Image.fromarray(result).save("scratch.jpg")
```

### Lines, texture, and a soft rim

These keywords apply to `algorithm="patchmatch"`. Passing any of them with `algorithm="philippeau"` raises `UnsupportedOption`, including when the value is the usual default. An unknown keyword raises `TypeError`.

Turn on structure matching when the hole crosses an edge or a repeating texture. `edge_weighted` matches gradients as well as color. `reverse_search` pulls in source patches the first pass underused. `hole_refine` runs a few extra full-resolution passes on the hole only.

```python
result = inpaint_pyramid(
    image,
    mask,
    seed=42,
    edge_weighted=True,
    reverse_search=True,
    hole_refine=2,
)
```

The default already blends a narrow band along the hole rim (`poisson_blend=True`). Widen that band when the seam is still visible. `mask_feather` is a width in pixels and replaces the default rim blend.

```python
result = inpaint_pyramid(image, mask, seed=42, mask_feather=8.0)
```

`patch_size` is the square patch side and must be odd. A larger side sees more context and costs more. `em_loops` repeats match and vote on the coarse pyramid levels.

```python
result = inpaint_pyramid(image, mask, patch_size=7, em_loops=2, seed=42)
```

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | `5` | Odd positive int; square patch side for matching and voting. |
| `seed` | `None` | RNG seed. `None` leaves the default solver unseeded. |
| `solver_class` | `None` | `None` uses `CythonSolver` when the extension is available, otherwise `PythonSolver`. |
| `em_loops` | `1` | Outer match→paint→rematch loops. On pyramids, applied at coarse levels only (finest stays `1`). |
| `edge_weighted` | `False` | `False` = color-only matching. `True` = color + boosted gradient channels. Hole-touching gradients are suppressed so zeroed holes do not create fake edges. |
| `reverse_search` | `False` | After each EM loop, try assigning underused near-hole sources to better queries (completeness). |
| `poisson_blend` | `True` | Soften seams by blending the PatchMatch fill with a harmonic membrane near the hole rim. Ignored when `mask_feather > 0` (feather supplies the band). |
| `hole_refine` | `0` | Extra full-resolution PatchMatch iterations after the main solve (`0` disables). Only hole pixels are painted; the user mask is not modified. |
| `mask_feather` | `0.0` | Soft compose width in pixels from the hole boundary. Uses a distance ramp toward a harmonic rim pull (not the original hole content). Does **not** change the mask used for matching. |

`inpaint_single` takes the same keywords, plus `iterations` (default `5`).

Masks may be `bool`, `{0,1}`, or `{0,255}`. The library does not dilate or erode the mask used for matching. `mask_feather` only changes the final composite.

### Xavier Philippeau

`algorithm="philippeau"`, or `inpaint_philippeau`, runs Xavier Philippeau's pyramid: bidirectional nearest-neighbor fields, a texture-energy channel, split votes on the fine levels, and a short border blend. The implementation is Cython and does not use OpenCV. Known pixels stay exact. Passing a PatchMatch keyword (`solver_class`, `em_loops`, `edge_weighted`, `reverse_search`, `poisson_blend`, `hole_refine`, `mask_feather`, or `iterations`) raises `UnsupportedOption`.

In this mode `patch_size` is a **radius**: each patch spans `2 * patch_size + 1` pixels. Any integer from 1 up is valid, including even values. A radius of 3 is a 7×7 patch. `inpaint_philippeau` defaults to radius 15. `inpaint_pyramid` and `inpaint_single` still default `patch_size` to 5, and that 5 is a radius here.

Pass `seed` for a repeatable fill. If it is omitted, a 32-bit seed is drawn for that run.

`inpaint_philippeau` requires an HxWx3 `uint8` image. `inpaint_pyramid` and `inpaint_single` round other numeric images into `uint8` and clip them to 0–255 before filling. The same `uint8` image, mask, radius, and seed produce the same fill from either call.

Fill a photo. Radius 7 is a 15×15 patch, a practical size for a region a few dozen pixels across. Radius 15 is the default on `inpaint_philippeau` and suits a larger hole.

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_philippeau

image = np.array(Image.open("photo.jpg").convert("RGB"))
mask = np.array(Image.open("hole.png").convert("L")) > 0

result = inpaint_philippeau(image, mask, patch_size=7, seed=42)
Image.fromarray(result).save("filled.jpg")
```

`inpaint_pyramid(..., algorithm="philippeau")` is the same fill when the image is already `uint8` and `patch_size` and `seed` match.

```python
from patchmatch_cython import inpaint_pyramid

result = inpaint_pyramid(image, mask, patch_size=7, seed=42, algorithm="philippeau")
```

If `mask` is omitted, pure white pixels (`255, 255, 255`) are the holes. Use that for a plate or sky that was painted white.

```python
result = inpaint_philippeau(image, patch_size=7, seed=42)
```

`global_mask` marks pixels that must not be filled and must not be copied from, such as a logo or a frame. `ijmap` is an optional HxWx3 `float32` regularity map; the first two channels are read, and `guide_weight` (default `0.25`) sets its weight. `inpaint_pyramid` and `inpaint_single` do not take these arguments.

```python
keep_out = np.zeros(image.shape[:2], dtype=bool)
keep_out[0:40, 0:120] = True  # logo in the corner

result = inpaint_philippeau(
    image,
    mask,
    global_mask=keep_out,
    patch_size=7,
    seed=42,
)
```

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
