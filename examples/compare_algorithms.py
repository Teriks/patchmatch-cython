#!/usr/bin/env python
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

"""Compare the Barnes PatchMatch pyramid with the Philippeau pyramid.

With no arguments, the script uses the same test image and mask as
``solver_speed_comparison.py``. Pass image and mask paths in pairs to compare real
files. The mask is a single channel; any non-zero pixel is a hole.

PatchMatch ``patch_size`` is the odd side length of the patch. Philippeau
``patch_size`` is a radius, so the patch side is ``2 * radius + 1``.

Usage:
    python examples/compare_algorithms.py
    python examples/compare_algorithms.py photo.jpg hole.png
    python examples/compare_algorithms.py photo.jpg hole.png other.jpg other.png
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from patchmatch_cython import inpaint_philippeau, inpaint_pyramid
from solver_speed_comparison import create_test_image, create_test_mask


def load_pair(image_path, mask_path):
    """Load an RGB image and a boolean hole mask."""
    from PIL import Image

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask {mask_path} is {mask.shape}, image {image_path} is {image.shape[:2]}"
        )
    return image, mask


def timed(label, function):
    print(f"  {label}...", end=" ", flush=True)
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    print(f"{elapsed:.2f}s")
    return result, elapsed


def compare_one(name, image, mask, patch_size, radius, seed):
    print(f"{name}: {image.shape[1]}x{image.shape[0]}, {int(np.count_nonzero(mask))} hole pixels")
    patchmatch, patch_time = timed(
        f"PatchMatch (patch {patch_size})",
        lambda: inpaint_pyramid(image, mask, patch_size=patch_size, seed=seed),
    )
    philippeau, phil_time = timed(
        f"Philippeau (radius {radius})",
        lambda: inpaint_philippeau(image, mask, patch_size=radius, seed=seed),
    )
    return {
        "name": name,
        "image": image,
        "mask": mask,
        "patch_size": patch_size,
        "radius": radius,
        "results": {
            "PatchMatch": {"time": patch_time, "image": patchmatch},
            "Philippeau": {"time": phil_time, "image": philippeau},
        },
    }


def save_figure(records, output):
    """Quality sheet: source, mask, and both fills, with runtime underneath."""
    algorithms = ["PatchMatch", "Philippeau"]
    colors = {"PatchMatch": "#1565c0", "Philippeau": "#e57373"}
    shown = records[0]
    image_height, image_width = shown["image"].shape[:2]

    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.35, 0.7], hspace=0.45, wspace=0.25)

    ax_orig = fig.add_subplot(grid[0, 0])
    ax_mask = fig.add_subplot(grid[0, 1])
    ax_orig.imshow(shown["image"])
    ax_orig.set_title(f"Original ({image_height}x{image_width})")
    ax_orig.axis("off")
    ax_mask.imshow(shown["mask"], cmap="gray")
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    for column, name in enumerate(algorithms, start=2):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(shown["results"][name]["image"])
        elapsed = shown["results"][name]["time"]
        if name == "PatchMatch":
            detail = f"patch {shown['patch_size']}, {elapsed:.2f}s"
        else:
            side = 2 * shown["radius"] + 1
            detail = f"radius {shown['radius']} ({side}x{side}), {elapsed:.2f}s"
        axis.set_title(f"{name}\n{detail}")
        axis.axis("off")

    ax_runtime = fig.add_subplot(grid[1, :])
    x_pos = np.arange(len(records))
    bar_width = 0.8 / len(algorithms)
    for index, name in enumerate(algorithms):
        offset = (index - (len(algorithms) - 1) / 2) * bar_width
        values = [record["results"][name]["time"] for record in records]
        bars = ax_runtime.bar(x_pos + offset, values, bar_width, label=name, color=colors[name])
        for bar in bars:
            bar_height = bar.get_height()
            if bar_height > 0:
                ax_runtime.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar_height,
                    f"{bar_height:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax_runtime.set_xticks(x_pos)
    ax_runtime.set_xticklabels([record["name"] for record in records])
    ax_runtime.set_ylim(0, 5)
    ax_runtime.set_ylabel("Time (seconds)")
    ax_runtime.set_title("Runtime")
    ax_runtime.legend(fontsize=8)
    ax_runtime.grid(True, alpha=0.3)

    fig.suptitle("Inpaint quality — PatchMatch and Philippeau", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"SUCCESS: Results saved to '{output}'")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PatchMatch and Philippeau inpainting on the same holes.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Image and mask paths in pairs. Omit them to use the speed-comparison test image.",
    )
    parser.add_argument("--patch-size", type=int, default=5, help="PatchMatch patch side (odd, default 5)")
    parser.add_argument("--radius", type=int, default=7, help="Philippeau patch radius (default 7, a 15x15 patch)")
    parser.add_argument("--seed", type=int, default=42, help="Seed shared by both algorithms")
    parser.add_argument("--size", type=int, default=512, help="Side length of the shared test image")
    parser.add_argument("--output", default="algo_comparison.png", help="Where to write the figure")
    args = parser.parse_args()
    if len(args.paths) % 2 != 0:
        parser.error("pass image and mask paths in pairs")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        parser.error("--patch-size must be a positive odd integer")
    if args.radius < 1:
        parser.error("--radius must be a positive integer")
    return args


def scenes_from_args(args):
    if not args.paths:
        image = create_test_image((args.size, args.size), args.seed)
        mask = create_test_mask(image.shape, args.seed)
        return [("test", image, mask)]
    scenes = []
    for index in range(0, len(args.paths), 2):
        image_path = Path(args.paths[index])
        mask_path = Path(args.paths[index + 1])
        scenes.append((image_path.name, *load_pair(image_path, mask_path)))
    return scenes


def main():
    args = parse_args()
    print("Algorithm comparison")
    print("-" * 50)
    records = [
        compare_one(name, image, mask, args.patch_size, args.radius, args.seed)
        for name, image, mask in scenes_from_args(args)
    ]
    save_figure(records, args.output)
    return True


if __name__ == "__main__":
    try:
        ok = main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    if not ok:
        raise SystemExit(1)
