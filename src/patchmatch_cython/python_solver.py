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

import typing

import numpy as np

from .imageops import (
    as_bool_mask,
    build_feature_image,
    build_query_patch_centers,
    build_source_ring_centers,
    build_spatial_vote_weights,
    build_valid_patch_centers,
    sanitize_nnf,
)
from .solver import Solver

Coords = typing.Union[typing.Tuple[int, int], typing.List[int], np.ndarray]

# Gradient channel scale when edge_weighted=True.
_EDGE_WEIGHT_BOOST = 3.0
# Stronger completeness when reverse_search=True.
_REVERSE_USAGE_PENALTY = 24.0
_DEFAULT_USAGE_PENALTY = 8.0


class PythonSolver(Solver):
    """
    Python implementation of the PatchMatch algorithm for image inpainting.

    This class provides a pure Python implementation of the PatchMatch algorithm,
    inheriting from the base :class:`Solver` class. It performs image inpainting
    by finding approximate nearest neighbors between patches in the source image
    and reconstructing the masked regions.

    The algorithm works by iteratively improving a nearest neighbor field (NNF)
    through propagation and random search steps, alternating between forward
    and backward passes for better convergence.

    :param image: Input image with masked regions to be inpainted
    :type image: numpy.ndarray
    :param mask: Binary mask indicating regions to inpaint (True for masked pixels)
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches used for matching (must be odd)
    :type patch_size: int
    :param edge_weighted: If True, match with boosted gradient channels; if False,
                          use color-only features
    :type edge_weighted: bool
    """

    def __init__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            patch_size: int,
            edge_weighted: bool = False,
    ) -> None:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")

        mask_bool = as_bool_mask(mask)
        image_f32 = np.ascontiguousarray(image, dtype=np.float32)
        if image_f32.ndim == 2:
            image_f32 = image_f32[:, :, None]
        if image_f32.ndim != 3:
            raise ValueError("image must have shape (H, W) or (H, W, C)")
        if mask_bool.shape != image_f32.shape[:2]:
            raise ValueError("mask shape must match image spatial dimensions")
        if np.all(mask_bool):
            raise ValueError("mask covers the entire image; no source pixels available")

        super().__init__(image_f32, mask_bool, patch_size)

        self.edge_weighted: bool = bool(edge_weighted)
        # Off = color-only; on = boosted gradients. Never use unmasked 1.0 grads
        # from a zeroed hole (that created fake edges at the hole boundary).
        self._edge_weight: float = _EDGE_WEIGHT_BOOST if self.edge_weighted else 0.0

        self._nnf: np.ndarray | None = None
        self._nn_dist: np.ndarray | None = None
        self._usage: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self.mask = mask_bool
        self._masked_coords: np.ndarray = np.array(np.where(self.mask)).T
        self._query_mask: np.ndarray = build_query_patch_centers(
            self.mask, self.half_patch_size
        )
        self._query_coords: np.ndarray = np.array(np.where(self._query_mask)).T
        self._is_valid_coords: np.ndarray = build_valid_patch_centers(
            self.mask, self.half_patch_size
        )
        self._valid_coords: np.ndarray = np.array(np.where(self._is_valid_coords)).T
        if len(self._valid_coords) == 0:
            raise ValueError("no valid source patch centers available for matching")

        self._ring_mask: np.ndarray = build_source_ring_centers(
            self.mask, self.half_patch_size
        )
        self._ring_coords: np.ndarray = np.array(np.where(self._ring_mask)).T

        self._spatial_weights: np.ndarray = build_spatial_vote_weights(self.half_patch_size)
        self._source_features: np.ndarray = build_feature_image(
            self.image, edge_weight=self._edge_weight, mask=self.mask
        )
        patch_span = float(self.patch_size * self.patch_size)
        feat_channels = float(self._source_features.shape[2])
        self._dist_norm: float = max(feat_channels * patch_span * 25.0, 1.0)
        self._usage_penalty: float = _DEFAULT_USAGE_PENALTY

    @property
    def nnf(self) -> np.ndarray:
        return self._nnf

    @nnf.setter
    def nnf(self, value: np.ndarray):
        self._nnf = value

    def _features(self, image: np.ndarray, *, source_side: bool = False) -> np.ndarray:
        # Source features must mask hole-touching gradients. Query features are
        # built from the current estimate (hole already painted) so leave mask off.
        return build_feature_image(
            image,
            edge_weight=self._edge_weight,
            mask=self.mask if source_side else None,
        )

    def _bounds_check_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_patch(self, image: np.ndarray, coords: Coords) -> np.ndarray:
        x, y = coords
        h = self.half_patch_size
        y1, y2 = max(0, y - h), min(image.shape[0], y + h + 1)
        x1, x2 = max(0, x - h), min(image.shape[1], x + h + 1)
        patch = image[y1:y2, x1:x2]
        pad_top = max(0, h - y)
        pad_bottom = max(0, y + h + 1 - image.shape[0])
        pad_left = max(0, h - x)
        pad_right = max(0, x + h + 1 - image.shape[1])
        if pad_top or pad_bottom or pad_left or pad_right:
            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
        return patch

    def _patch_distance(
            self,
            patch_image1: np.ndarray, patch_coords1: Coords,
            patch_image2: np.ndarray, patch_coords2: Coords
    ) -> float:
        patch1 = self._get_patch(patch_image1, patch_coords1)
        patch2 = self._get_patch(patch_image2, patch_coords2)
        return float(np.sum((patch1 - patch2) ** 2))

    def _candidate_score(
            self,
            features: np.ndarray,
            y: int,
            x: int,
            cand_x: int,
            cand_y: int,
    ) -> float:
        dist = self._patch_distance(features, [x, y], self._source_features, [cand_x, cand_y])
        if self._usage is not None:
            dist += self._usage_penalty * float(self._usage[cand_y, cand_x])
        return dist

    def _sanitize_nnf(self) -> None:
        if self._nnf is None:
            return
        self._nnf = sanitize_nnf(
            self._nnf, self._query_mask, self._is_valid_coords, rng=self._rng
        )

    def _randomize_nnf(self) -> None:
        h, w = self.height, self.width
        self._nnf = np.zeros((h, w, 2), dtype=np.int32)
        self._nn_dist = np.full((h, w), np.float32(1e20), dtype=np.float32)
        self._usage = np.zeros((h, w), dtype=np.float32)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        self._nnf[..., 0] = x_coords
        self._nnf[..., 1] = y_coords

        if len(self._query_coords) > 0:
            random_indices = self._rng.integers(0, len(self._valid_coords), size=len(self._query_coords))
            random_coords = self._valid_coords[random_indices]
            query_y, query_x = self._query_coords[:, 0], self._query_coords[:, 1]
            self._nnf[query_y, query_x] = random_coords[:, ::-1]

    def _initialize_distances(self, features: np.ndarray) -> None:
        assert self._nnf is not None and self._nn_dist is not None
        for y, x in self._query_coords:
            mx, my = self._nnf[y, x]
            self._nn_dist[y, x] = np.float32(
                self._patch_distance(features, [x, y], self._source_features, [mx, my])
            )

    def _paint_with_nnf(self, target: np.ndarray) -> None:
        if len(self._masked_coords) == 0 or self._nnf is None:
            return

        half = self.half_patch_size
        acc = np.zeros_like(target, dtype=np.float64)
        weight = np.zeros(self.mask.shape, dtype=np.float64)
        spatial = self._spatial_weights
        nn_dist = self._nn_dist

        query_y, query_x = self._query_coords[:, 0], self._query_coords[:, 1]
        matches = self._nnf[query_y, query_x]
        match_x, match_y = matches[:, 0], matches[:, 1]

        for i in range(len(self._query_coords)):
            cy = int(query_y[i])
            cx = int(query_x[i])
            my = int(match_y[i])
            mx = int(match_x[i])
            dist = float(nn_dist[cy, cx]) if nn_dist is not None else 0.0
            sim = 1.0 / (1.0 + dist / self._dist_norm)

            for dy in range(-half, half + 1):
                ty = cy + dy
                sy = my + dy
                if ty < 0 or ty >= self.height or sy < 0 or sy >= self.height:
                    continue
                for dx in range(-half, half + 1):
                    tx = cx + dx
                    sx = mx + dx
                    if tx < 0 or tx >= self.width or sx < 0 or sx >= self.width:
                        continue
                    if not self.mask[ty, tx]:
                        continue
                    w = float(spatial[dy + half, dx + half]) * sim
                    acc[ty, tx] += self.image[sy, sx] * w
                    weight[ty, tx] += w

        paintable = self.mask & (weight > 0)
        if np.any(paintable):
            target[paintable] = (
                acc[paintable] / weight[paintable][:, None]
            ).astype(target.dtype, copy=False)

        missing = self.mask & (weight == 0)
        if np.any(missing):
            my, mx = np.where(missing)
            fallback = self._nnf[my, mx]
            target[my, mx] = self.image[fallback[:, 1], fallback[:, 0]]

    def _propagate_and_search(self, features: np.ndarray, it: int) -> None:
        assert self._nnf is not None and self._nn_dist is not None and self._usage is not None
        self._usage.fill(0)
        max_search_radius = max(self.height, self.width)
        neighbor_offsets = [np.array([(-1, 0), (0, -1)]), np.array([(1, 0), (0, 1)])]
        is_forward = it % 2 == 0
        coords_order = self._query_coords if is_forward else reversed(self._query_coords)
        neighbors = neighbor_offsets[0] if is_forward else neighbor_offsets[1]

        for y, x in coords_order:
            best_match = self._nnf[y, x]
            bx, by = int(best_match[0]), int(best_match[1])
            best_distance = self._candidate_score(features, y, x, bx, by)

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if not self._bounds_check_coord(nx, ny):
                    continue
                candidate = self._nnf[ny, nx] - np.array([dx, dy])
                cx, cy = int(candidate[0]), int(candidate[1])
                if self._bounds_check_coord(cx, cy) and self._is_valid_coords[cy, cx]:
                    distance = self._candidate_score(features, y, x, cx, cy)
                    if distance < best_distance:
                        best_match = candidate
                        best_distance = distance
                        bx, by = cx, cy

            search_radius = max_search_radius
            while search_radius > 1:
                rx = bx + int(self._rng.integers(-search_radius, search_radius + 1))
                ry = by + int(self._rng.integers(-search_radius, search_radius + 1))
                if self._bounds_check_coord(rx, ry) and self._is_valid_coords[ry, rx]:
                    distance = self._candidate_score(features, y, x, rx, ry)
                    if distance < best_distance:
                        best_match = np.array([rx, ry])
                        best_distance = distance
                        bx, by = rx, ry
                search_radius //= 2

            self._nnf[y, x] = best_match
            self._nn_dist[y, x] = np.float32(
                self._patch_distance(
                    features, [x, y], self._source_features, [bx, by]
                )
            )
            self._usage[by, bx] += 1.0

    def _reverse_search(self, features: np.ndarray) -> None:
        """Try assigning underused near-hole sources to better-matching queries."""
        if (
            self._nnf is None
            or self._nn_dist is None
            or self._usage is None
            or len(self._ring_coords) == 0
            or len(self._query_coords) == 0
        ):
            return

        n_query = len(self._query_coords)
        n_samples = min(24, max(8, n_query // 4))
        neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for sy, sx in self._ring_coords:
            best_qy = -1
            best_qx = -1
            best_d = 1e30

            for _ in range(n_samples):
                idx = int(self._rng.integers(0, n_query))
                qy = int(self._query_coords[idx, 0])
                qx = int(self._query_coords[idx, 1])
                d = self._patch_distance(
                    features, [qx, qy], self._source_features, [sx, sy]
                )
                if d < best_d:
                    best_d = d
                    best_qy, best_qx = qy, qx

            # Probe query-neighbors of the best sample (matches Cython).
            if best_qy >= 0:
                for qdy, qdx in neighbor_offsets:
                    qy = best_qy + qdy
                    qx = best_qx + qdx
                    if not self._bounds_check_coord(qx, qy):
                        continue
                    if not self._query_mask[qy, qx]:
                        continue
                    d = self._patch_distance(
                        features, [qx, qy], self._source_features, [sx, sy]
                    )
                    if d < best_d:
                        best_d = d
                        best_qy, best_qx = qy, qx

            if best_qy < 0:
                continue

            current = float(self._nn_dist[best_qy, best_qx])
            if best_d >= current:
                continue

            old_mx = int(self._nnf[best_qy, best_qx, 0])
            old_my = int(self._nnf[best_qy, best_qx, 1])
            if self._usage[old_my, old_mx] > 0:
                self._usage[old_my, old_mx] -= 1.0
            self._nnf[best_qy, best_qx, 0] = sx
            self._nnf[best_qy, best_qx, 1] = sy
            self._nn_dist[best_qy, best_qx] = np.float32(best_d)
            self._usage[sy, sx] += 1.0

    def __call__(
            self,
            iterations: int = 5,
            randomize_nnf: bool = True,
            nnf: typing.Optional[np.ndarray] = None,
            seed: typing.Optional[int] = None,
            em_loops: int = 1,
            reverse_search: bool = False,
    ) -> np.ndarray:
        if em_loops < 1:
            raise ValueError("em_loops must be >= 1")

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._usage_penalty = (
            _REVERSE_USAGE_PENALTY if reverse_search else _DEFAULT_USAGE_PENALTY
        )

        inpainted_image = self.image.copy()
        inpainted_features = self._features(inpainted_image)

        if randomize_nnf:
            self._randomize_nnf()
        else:
            if nnf is None:
                raise ValueError("nnf must be provided when randomize_nnf is False")
            self._nnf = np.ascontiguousarray(nnf, dtype=np.int32)
            self._nn_dist = np.full((self.height, self.width), np.float32(1e20), dtype=np.float32)
            self._usage = np.zeros((self.height, self.width), dtype=np.float32)
            self._sanitize_nnf()

        self._initialize_distances(inpainted_features)
        self._paint_with_nnf(inpainted_image)
        inpainted_features = self._features(inpainted_image)

        for em in range(em_loops):
            for it in range(iterations):
                self._propagate_and_search(inpainted_features, it)
                self._paint_with_nnf(inpainted_image)
                inpainted_features = self._features(inpainted_image)

            if reverse_search:
                self._reverse_search(inpainted_features)
                self._paint_with_nnf(inpainted_image)
                inpainted_features = self._features(inpainted_image)

        return inpainted_image
