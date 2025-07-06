# PatchMatch Cython

High-performance PatchMatch implementation for pyramidical image inpainting in cython.

This package can function without `python-opencv` / `python-opencv-headless` with only slightly reduced quality.

Installing OpenCV results in this packaging using it for upsampling (`INTER_LINEAR`).

When OpenCV is not installed, an implementation of bilinear upscaling using numpy will be used.

This package provides a pure python solver and a cython solver, the cython solver is 
around 30 to 50 times faster than the python solver depending on patch size.

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

# Load image and mask
image = cv2.imread('image.jpg')
mask = cv2.imread('mask.png', 0) > 128

# Inpaint
result = inpaint_pyramid(image, mask)

cv2.imwrite('result.jpg', result)
```

### With Pillow

```python
import numpy as np
from PIL import Image
from patchmatch_cython import inpaint_pyramid

# Load image and mask  
image = np.array(Image.open('image.jpg'))
mask = np.array(Image.open('mask.png').convert('L')) > 128

# Inpaint (auto-selects fastest available solver)
result = inpaint_pyramid(image, mask)

Image.fromarray(result).save('result.jpg')
```

### Explicit Solver Selection

```python
from patchmatch_cython import PythonSolver, CythonSolver, inpaint_pyramid

# Force specific solver
result = inpaint_pyramid(image, mask, solver_class=CythonSolver)  # Fast
result = inpaint_pyramid(image, mask, solver_class=PythonSolver)  # Fallback
```

## Package Extras

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

Install build dependencies:

```bash
pip install cibuildwheel
```

### Build Wheels

Use the included `local_build.py` script to build locally:

```bash
# Build for all supported Python versions (3.10-3.13)
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

### Build Options

- `--python`: Specify Python versions to build for (e.g., `3.11 3.12`)
- `--platform`: Target platform (`auto`, `linux`, `macos`, `windows`)
- `--test`: Test wheel installation after building
- `--no-clean`: Skip cleaning build artifacts before building


## Requirements

- Python 3.10+
- NumPy (auto-installed)
- C++ compiler (for building from source)