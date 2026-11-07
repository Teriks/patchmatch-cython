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

import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from patchmatch_cython import PythonSolver
    PYTHON_AVAILABLE = True
except ImportError:
    print("Python solver not available")
    PYTHON_AVAILABLE = False

try:
    from patchmatch_cython import CythonSolver
    CYTHON_AVAILABLE = True
except ImportError:
    print("Cython solver not available")
    CYTHON_AVAILABLE = False

from patchmatch_cython.imageops import opencv_available, set_use_opencv


def create_test_image(size=(512, 512), seed: int = 42) -> np.ndarray:
    """Create a synthetic test image with various features"""
    rng = np.random.default_rng(seed)
    h, w = size

    yy = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[..., 0] = yy
    image[..., 1] = xx

    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
        draw.ellipse([260, 60, 340, 140], fill=(0, 255, 0))
        draw.ellipse([140, 270, 260, 330], fill=(0, 0, 255))
        for _ in range(5):
            pt1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            pt2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            color = tuple(int(v) for v in rng.integers(100, 255, size=3))
            draw.line([pt1, pt2], fill=color, width=2)
        image_array = np.asarray(pil, dtype=np.int32)
    except ImportError:
        image_array = image.astype(np.int32)
        image_array[50:150, 50:150] = (255, 0, 0)
        y, x = np.ogrid[:h, :w]
        circle_mask = (x - 300) ** 2 + (y - 100) ** 2 <= 40 ** 2
        image_array[circle_mask] = (0, 255, 0)

    noise = rng.integers(0, 30, size=(h, w, 3))
    return np.clip(image_array + noise, 0, 255).astype(np.uint8)


def create_test_mask(image_shape, seed: int = 42) -> np.ndarray:
    """Create a test mask with multiple holes"""
    np.random.seed(seed)
    h, w = image_shape[:2]

    try:
        from PIL import Image, ImageDraw

        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        draw.ellipse(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            fill=255,
        )
        for _ in range(3):
            pt1 = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))
            pt2 = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))
            draw.line([pt1, pt2], fill=255, width=10)
        return np.array(mask_img) > 0

    except ImportError:
        mask = np.zeros((h, w), dtype=bool)
        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask |= (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        for _ in range(3):
            y_start = np.random.randint(h // 4, 3 * h // 4)
            x_start = np.random.randint(w // 4, 3 * w // 4)
            y_end = np.random.randint(h // 4, 3 * h // 4)
            x_end = np.random.randint(w // 4, 3 * w // 4)
            steps = max(abs(x_end - x_start), abs(y_end - y_start))
            if steps > 0:
                for step in range(steps):
                    t = step / steps
                    yy = int(y_start + t * (y_end - y_start))
                    xx = int(x_start + t * (x_end - x_start))
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            ny, nx = yy + dy, xx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                mask[ny, nx] = True
        return mask


def benchmark_solver(
        solver_class,
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        seed: int = 42,
        runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark a solver through ``inpaint_pyramid``."""
    from patchmatch_cython import inpaint_pyramid

    times = []
    result_image = None

    for run in range(runs):
        print(f"  Run {run + 1}/{runs}...", end=" ")
        start_time = time.time()
        result = inpaint_pyramid(
            image=image.copy(),
            mask=mask.copy(),
            patch_size=patch_size,
            solver_class=solver_class,
            seed=seed,
        )
        runtime = time.time() - start_time
        times.append(runtime)
        print(f"{runtime:.3f}s")
        if run == 0:
            result_image = result.copy()

    return {
        'times': times,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'result_image': result_image,
    }


def create_comprehensive_visualization(
        all_results: Dict[str, Dict[int, Dict[str, Any]]],
        image_size: Tuple[int, int],
) -> None:
    """Create visualization comparing solvers and OpenCV vs NumPy backends."""
    backends = list(all_results.keys())
    patch_sizes = list(next(iter(all_results.values())).keys())
    solver_names = []
    for backend in backends:
        for patch_size in patch_sizes:
            for name in all_results[backend][patch_size]:
                if name not in solver_names:
                    solver_names.append(name)

    n_result_cols = max(len(backends) * len(solver_names), 1)
    fig = plt.figure(figsize=(4 + 3 * n_result_cols, 8.5))
    gs = fig.add_gridspec(
        4, 2 + n_result_cols,
        height_ratios=[1.15, 1.05, 0.9, 1.0],
        hspace=0.45,
        wspace=0.35,
    )

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_mask = fig.add_subplot(gs[0, 1])
    image = create_test_image(image_size)
    mask = create_test_mask(image.shape)
    ax_orig.imshow(image)
    ax_orig.set_title(f'Original ({image_size[0]}x{image_size[1]})')
    ax_orig.axis('off')
    ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title('Mask')
    ax_mask.axis('off')

    col = 2
    first_patch = patch_sizes[0]
    for backend in backends:
        for solver_name in solver_names:
            if solver_name not in all_results[backend][first_patch]:
                continue
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(all_results[backend][first_patch][solver_name]['result_image'])
            ax.set_title(f'{solver_name}\n{backend}\npatch {first_patch}')
            ax.axis('off')
            col += 1

    # Grouped runtime bars: one group per patch size, bars for each backend×solver.
    ax_perf = fig.add_subplot(gs[1, :])
    series_labels = []
    series_data = []
    colors = []
    palette = {
        ('NumPy', 'Python'): '#e57373',
        ('NumPy', 'Cython'): '#64b5f6',
        ('OpenCV', 'Python'): '#c62828',
        ('OpenCV', 'Cython'): '#1565c0',
    }
    for backend in backends:
        for solver_name in solver_names:
            values = []
            present = False
            for patch_size in patch_sizes:
                stats = all_results[backend][patch_size].get(solver_name)
                if stats is None:
                    values.append(0.0)
                else:
                    values.append(stats['mean_time'])
                    present = True
            if present:
                series_labels.append(f'{solver_name} ({backend})')
                series_data.append(values)
                colors.append(palette.get((backend, solver_name), '#9e9e9e'))

    x_pos = np.arange(len(patch_sizes))
    width = 0.8 / max(len(series_data), 1)
    for i, (label, values) in enumerate(zip(series_labels, series_data)):
        offset = (i - (len(series_data) - 1) / 2) * width
        bars = ax_perf.bar(
            x_pos + offset, values, width, label=label, color=colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_perf.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{height:.2f}s',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    ax_perf.set_xticks(x_pos)
    ax_perf.set_xticklabels([f'Patch {p}' for p in patch_sizes])
    ax_perf.set_ylabel('Time (seconds)')
    ax_perf.set_title('Performance: solvers × imageops backend')
    ax_perf.legend(fontsize=8)
    ax_perf.grid(True, alpha=0.3)

    # OpenCV speedup vs NumPy for each solver/patch
    ax_speedup = fig.add_subplot(gs[2, :])
    if 'OpenCV' in all_results and 'NumPy' in all_results:
        speedup_labels = []
        speedup_values = []
        for patch_size in patch_sizes:
            for solver_name in solver_names:
                np_stats = all_results['NumPy'][patch_size].get(solver_name)
                cv_stats = all_results['OpenCV'][patch_size].get(solver_name)
                if not np_stats or not cv_stats or cv_stats['mean_time'] <= 0:
                    continue
                speedup_labels.append(f'{solver_name}\npatch {patch_size}')
                speedup_values.append(np_stats['mean_time'] / cv_stats['mean_time'])
        bars = ax_speedup.bar(speedup_labels, speedup_values, color='#81c784')
        for bar in bars:
            height = bar.get_height()
            ax_speedup.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.2f}x',
                ha='center',
                va='bottom',
            )
        ax_speedup.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax_speedup.set_ylabel('OpenCV speedup vs NumPy')
        ax_speedup.set_title('imageops OpenCV acceleration factor')
        ax_speedup.grid(True, alpha=0.3)
    else:
        ax_speedup.axis('off')
        ax_speedup.text(
            0.5, 0.5, 'OpenCV not installed — NumPy-only results',
            ha='center', va='center', transform=ax_speedup.transAxes,
        )

    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')
    summary = "Performance Summary\n" + "=" * 60 + "\n\n"
    for backend in backends:
        summary += f"[{backend} imageops]\n"
        for patch_size in patch_sizes:
            summary += f"  Patch {patch_size}:\n"
            for solver_name in solver_names:
                stats = all_results[backend][patch_size].get(solver_name)
                if not stats:
                    continue
                summary += (
                    f"    {solver_name}: "
                    f"{stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s\n"
                )
            py = all_results[backend][patch_size].get('Python')
            cy = all_results[backend][patch_size].get('Cython')
            if py and cy and cy['mean_time'] > 0:
                summary += f"    Cython/Python: {py['mean_time'] / cy['mean_time']:.1f}x\n"
        summary += "\n"
    ax_stats.text(
        0.02, 0.98, summary, transform=ax_stats.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
    )

    plt.suptitle(
        f'PatchMatch Performance — {image_size[0]}x{image_size[1]} '
        f'(OpenCV vs NumPy imageops)',
        fontsize=14,
        y=0.995,
    )
    plt.savefig('speed_comparison_result.png', dpi=150, bbox_inches='tight')
    print("SUCCESS: Results saved to 'speed_comparison_result.png'")
    plt.show()


def run_comprehensive_comparison(
        image_size: Tuple[int, int] = (512, 512),
        patch_sizes: List[int] = [5, 7],
        runs: int = 3,
        seed: int = 42,
) -> None:
    """Run comparison across patch sizes, solvers, and OpenCV on/off."""
    print(f"Running comprehensive comparison - {image_size[0]}x{image_size[1]} image")
    print("=" * 60)

    available_solvers = []
    if PYTHON_AVAILABLE:
        available_solvers.append(('Python', PythonSolver))
    if CYTHON_AVAILABLE:
        available_solvers.append(('Cython', CythonSolver))
    if not available_solvers:
        print("ERROR: No solvers available!")
        return

    backends = [('NumPy', False)]
    if opencv_available():
        backends.append(('OpenCV', True))
        print("OpenCV available: benchmarking NumPy and OpenCV imageops paths")
    else:
        print("OpenCV not installed: benchmarking NumPy imageops path only")

    print(f"Available solvers: {[name for name, _ in available_solvers]}")
    print(f"Testing patch sizes: {patch_sizes}")
    print(f"Runs per test: {runs}")
    print()

    image = create_test_image(image_size, seed)
    mask = create_test_mask(image.shape, seed)
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pixels to inpaint: {np.count_nonzero(mask)}")
    print()

    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    previous = set_use_opencv(True)

    try:
        for backend_name, use_opencv in backends:
            enabled = set_use_opencv(use_opencv)
            print(f"Backend: {backend_name} (HAS_OPENCV={enabled})")
            print("=" * 40)
            all_results[backend_name] = {}

            for patch_size in patch_sizes:
                print(f"Testing patch size {patch_size}")
                print("-" * 40)
                all_results[backend_name][patch_size] = {}

                for solver_name, solver_class in available_solvers:
                    print(f"Running {solver_name} solver...")
                    try:
                        results = benchmark_solver(
                            solver_class,
                            image,
                            mask,
                            patch_size=patch_size,
                            seed=seed,
                            runs=runs,
                        )
                        all_results[backend_name][patch_size][solver_name] = results
                        print(
                            f"  Average time: "
                            f"{results['mean_time']:.2f}s ± {results['std_time']:.2f}s"
                        )
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                print()
    finally:
        set_use_opencv(previous)

    create_comprehensive_visualization(all_results, image_size)


def main():
    print("PatchMatch Performance Comparison")
    print("=" * 50)

    if not PYTHON_AVAILABLE and not CYTHON_AVAILABLE:
        print("ERROR: No implementations available!")
        return

    # Two backends × two solvers × two patch sizes; keep runs modest.
    run_comprehensive_comparison(
        image_size=(512, 512),
        patch_sizes=[5, 7],
        runs=2,
        seed=42,
    )


if __name__ == "__main__":
    main()
