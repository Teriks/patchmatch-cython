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

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from patchmatch_cython.imageops import (
    build_feature_image as _build_feature_image_py,
    build_query_patch_centers as _build_query_patch_centers_py,
    build_source_ring_centers as _build_source_ring_centers_py,
    build_valid_patch_centers as _build_valid_patch_centers_py,
)

ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t ITYPE_t

cdef inline int _max_int(int a, int b) nogil:
    return a if a > b else b


cdef inline unsigned int _xorshift32(unsigned int *state) nogil:
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    if x == 0:
        x = 0xA5A5A5A5
    state[0] = x
    return x


cdef inline int _rand_int(unsigned int *state, int n) nogil:
    """Uniform integer in [0, n)."""
    if n <= 1:
        return 0
    return <int>(_xorshift32(state) % <unsigned int>n)


cdef np.ndarray _build_spatial_weights_py(int half_patch_size):
    if half_patch_size <= 0:
        return np.ones((1, 1), dtype=np.float32)
    cdef float sigma = half_patch_size / 2.0
    if sigma < 0.5:
        sigma = 0.5
    cdef np.ndarray offsets = np.arange(-half_patch_size, half_patch_size + 1, dtype=np.float32)
    cdef np.ndarray yy, xx
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    return np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)


cdef inline float _patch_distance_fast(
    DTYPE_t[:, :, :] img1,
    int y1,
    int x1,
    DTYPE_t[:, :, :] img2,
    int y2,
    int x2,
    int half_patch_size,
    int height,
    int width,
    int channels) nogil:

    cdef int dy, dx, c
    cdef int py1, px1, py2, px2
    cdef float dist = 0.0
    cdef float diff
    cdef float val1, val2

    for dy in range(-half_patch_size, half_patch_size + 1):
        for dx in range(-half_patch_size, half_patch_size + 1):
            py1 = y1 + dy
            px1 = x1 + dx
            py2 = y2 + dy
            px2 = x2 + dx
            for c in range(channels):
                if 0 <= py1 < height and 0 <= px1 < width:
                    val1 = img1[py1, px1, c]
                else:
                    val1 = 0.0
                if 0 <= py2 < height and 0 <= px2 < width:
                    val2 = img2[py2, px2, c]
                else:
                    val2 = 0.0
                diff = val1 - val2
                dist += diff * diff
    return dist


cdef class CythonSolverImpl:
    cdef:
        DTYPE_t[:, :, :] image
        DTYPE_t[:, :, :] source_features
        np.uint8_t[:, :] mask
        np.uint8_t[:, :] query_mask
        int patch_size, half_patch_size, height, width, channels, feat_channels
        int[:] _masked_y, _masked_x
        int _n_masked
        int[:] _query_y, _query_x
        int _n_query
        np.uint8_t[:, :] _is_valid_coords
        int[:] _valid_y, _valid_x
        int _n_valid
        int[:] _ring_y, _ring_x
        int _n_ring
        ITYPE_t[:, :, :] _nnf
        DTYPE_t[:, :] _nn_dist
        DTYPE_t[:, :] _usage
        DTYPE_t[:, :] _spatial_weights
        float _dist_norm
        float _usage_penalty
        float _edge_weight
        bint _edge_weighted
        unsigned int _rng_state

    def __init__(
            self,
            np.ndarray[DTYPE_t, ndim=3] image,
            np.ndarray[np.uint8_t, ndim=2] mask,
            int patch_size,
            bint edge_weighted=False,
    ):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")

        cdef np.ndarray mask_bool = (mask != 0).astype(np.uint8)
        if np.all(mask_bool):
            raise ValueError("mask covers the entire image; no source pixels available")

        cdef np.ndarray features
        cdef np.ndarray query
        cdef np.ndarray ring
        cdef np.ndarray spatial
        cdef float patch_span

        self._edge_weighted = edge_weighted
        # 0 = color-only; 3 = boosted structure. Mask zeros hole-touching grads.
        self._edge_weight = 3.0 if edge_weighted else 0.0
        features = _build_feature_image_py(
            image, edge_weight=self._edge_weight, mask=mask_bool.astype(bool)
        )

        self.image = image
        self.source_features = features
        self.mask = mask_bool
        self.patch_size = patch_size
        self.half_patch_size = patch_size // 2
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.channels = image.shape[2]
        self.feat_channels = features.shape[2]
        self._rng_state = 0xC0FFEE

        mask_indices = np.where(mask_bool)
        self._masked_y = np.asarray(mask_indices[0], dtype=np.int32)
        self._masked_x = np.asarray(mask_indices[1], dtype=np.int32)
        self._n_masked = len(self._masked_y)

        query = _build_query_patch_centers_py(mask_bool.astype(bool), self.half_patch_size).astype(np.uint8)
        self.query_mask = query
        query_indices = np.where(query)
        self._query_y = np.asarray(query_indices[0], dtype=np.int32)
        self._query_x = np.asarray(query_indices[1], dtype=np.int32)
        self._n_query = len(self._query_y)

        self._is_valid_coords = _build_valid_patch_centers_py(
            mask_bool.astype(bool), self.half_patch_size
        ).astype(np.uint8)
        valid_indices = np.where(self._is_valid_coords)
        self._valid_y = np.asarray(valid_indices[0], dtype=np.int32)
        self._valid_x = np.asarray(valid_indices[1], dtype=np.int32)
        self._n_valid = len(self._valid_y)
        if self._n_valid <= 0:
            raise ValueError("no valid source patch centers available for matching")

        ring = _build_source_ring_centers_py(
            mask_bool.astype(bool), self.half_patch_size
        ).astype(np.uint8)
        ring_indices = np.where(ring)
        self._ring_y = np.asarray(ring_indices[0], dtype=np.int32)
        self._ring_x = np.asarray(ring_indices[1], dtype=np.int32)
        self._n_ring = len(self._ring_y)

        spatial = _build_spatial_weights_py(self.half_patch_size)
        self._spatial_weights = spatial

        patch_span = <float>(patch_size * patch_size)
        self._dist_norm = self.feat_channels * patch_span * 25.0
        if self._dist_norm < 1.0:
            self._dist_norm = 1.0
        self._usage_penalty = 8.0

        self._nnf = np.zeros((self.height, self.width, 2), dtype=np.int32)
        self._nn_dist = np.full((self.height, self.width), np.float32(1e20), dtype=np.float32)
        self._usage = np.zeros((self.height, self.width), dtype=np.float32)

    @property
    def nnf(self) -> np.ndarray:
        return np.asarray(self._nnf)

    @nnf.setter
    def nnf(self, value: np.ndarray):
        if value.dtype != np.int32:
            value = value.astype(np.int32, copy=False)
        self._nnf = np.ascontiguousarray(value)

    cdef void _seed_rng(self, unsigned int seed) nogil:
        if seed == 0:
            seed = 0xA5A5A5A5
        self._rng_state = seed

    cpdef void _sanitize_nnf(self):
        cdef int i, y, x, match_x, match_y, rand_idx
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef np.uint8_t[:, :] is_valid = self._is_valid_coords
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef int[:] valid_y = self._valid_y
        cdef int[:] valid_x = self._valid_x
        cdef int height = self.height
        cdef int width = self.width
        cdef unsigned int rng = self._rng_state

        for i in range(self._n_query):
            y = query_y[i]
            x = query_x[i]
            match_x = nnf[y, x, 0]
            match_y = nnf[y, x, 1]
            if match_x < 0:
                match_x = 0
            elif match_x >= width:
                match_x = width - 1
            if match_y < 0:
                match_y = 0
            elif match_y >= height:
                match_y = height - 1
            if is_valid[match_y, match_x] == 0:
                rand_idx = _rand_int(&rng, self._n_valid)
                match_x = valid_x[rand_idx]
                match_y = valid_y[rand_idx]
            nnf[y, x, 0] = match_x
            nnf[y, x, 1] = match_y
        self._rng_state = rng

    cpdef void _randomize_nnf(self):
        cdef int i, y, x, rand_idx
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef DTYPE_t[:, :] nn_dist = self._nn_dist
        cdef DTYPE_t[:, :] usage = self._usage
        cdef int[:] valid_y = self._valid_y
        cdef int[:] valid_x = self._valid_x
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef unsigned int rng = self._rng_state

        for y in range(self.height):
            for x in range(self.width):
                nnf[y, x, 0] = x
                nnf[y, x, 1] = y
                nn_dist[y, x] = <DTYPE_t>1e20
                usage[y, x] = 0.0

        for i in range(self._n_query):
            y = query_y[i]
            x = query_x[i]
            rand_idx = _rand_int(&rng, self._n_valid)
            nnf[y, x, 0] = valid_x[rand_idx]
            nnf[y, x, 1] = valid_y[rand_idx]
        self._rng_state = rng

    cdef void _initialize_distances(self, DTYPE_t[:, :, :] features) nogil:
        cdef int i, y, x, match_x, match_y
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef DTYPE_t[:, :] nn_dist = self._nn_dist
        cdef DTYPE_t[:, :, :] source_features = self.source_features
        cdef int half_patch_size = self.half_patch_size
        cdef int height = self.height
        cdef int width = self.width
        cdef int feat_channels = self.feat_channels

        for i in range(self._n_query):
            y = query_y[i]
            x = query_x[i]
            match_x = nnf[y, x, 0]
            match_y = nnf[y, x, 1]
            nn_dist[y, x] = _patch_distance_fast(
                features, y, x, source_features, match_y, match_x,
                half_patch_size, height, width, feat_channels
            )

    cpdef void _paint_with_nnf(self, np.ndarray[DTYPE_t, ndim=3] target):
        cdef int i, y, x, match_x, match_y, c
        cdef int dy, dx, ty, tx, sy, sx
        cdef int half = self.half_patch_size
        cdef int height = self.height
        cdef int width = self.width
        cdef int channels = self.channels
        cdef float sim, w, dist
        cdef float dist_norm = self._dist_norm
        cdef DTYPE_t[:, :, :] source = self.image
        cdef DTYPE_t[:, :, :] target_view = target
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef DTYPE_t[:, :] nn_dist = self._nn_dist
        cdef DTYPE_t[:, :] spatial = self._spatial_weights
        cdef np.uint8_t[:, :] mask = self.mask
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef int[:] masked_y = self._masked_y
        cdef int[:] masked_x = self._masked_x
        cdef np.ndarray[np.float64_t, ndim=3] acc = np.zeros(
            (height, width, channels), dtype=np.float64
        )
        cdef np.ndarray[np.float64_t, ndim=2] weight = np.zeros(
            (height, width), dtype=np.float64
        )
        cdef np.float64_t[:, :, :] acc_view = acc
        cdef np.float64_t[:, :] weight_view = weight

        for i in range(self._n_query):
            y = query_y[i]
            x = query_x[i]
            match_x = nnf[y, x, 0]
            match_y = nnf[y, x, 1]
            dist = nn_dist[y, x]
            sim = 1.0 / (1.0 + dist / dist_norm)
            for dy in range(-half, half + 1):
                ty = y + dy
                sy = match_y + dy
                if ty < 0 or ty >= height or sy < 0 or sy >= height:
                    continue
                for dx in range(-half, half + 1):
                    tx = x + dx
                    sx = match_x + dx
                    if tx < 0 or tx >= width or sx < 0 or sx >= width:
                        continue
                    if mask[ty, tx] == 0:
                        continue
                    w = spatial[dy + half, dx + half] * sim
                    for c in range(channels):
                        acc_view[ty, tx, c] += source[sy, sx, c] * w
                    weight_view[ty, tx] += w

        for i in range(self._n_masked):
            y = masked_y[i]
            x = masked_x[i]
            if weight_view[y, x] > 0:
                for c in range(channels):
                    target_view[y, x, c] = <DTYPE_t>(acc_view[y, x, c] / weight_view[y, x])
            else:
                match_x = nnf[y, x, 0]
                match_y = nnf[y, x, 1]
                for c in range(channels):
                    target_view[y, x, c] = source[match_y, match_x, c]

    cdef void _propagate_and_search(self, DTYPE_t[:, :, :] inpainted_features, int iteration):
        cdef int idx, y, x, ny, nx
        cdef int best_x, best_y, cand_x, cand_y
        cdef float best_dist, dist, raw_dist
        cdef int search_radius, rx, ry
        cdef int height = self.height, width = self.width
        cdef int half_patch_size = self.half_patch_size
        cdef int feat_channels = self.feat_channels
        cdef float usage_penalty = self._usage_penalty
        cdef DTYPE_t[:, :, :] source_features = self.source_features
        cdef np.uint8_t[:, :] is_valid = self._is_valid_coords
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef DTYPE_t[:, :] nn_dist = self._nn_dist
        cdef DTYPE_t[:, :] usage = self._usage
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef unsigned int rng = self._rng_state
        cdef int start, end, step
        cdef int dx1, dy1, dx2, dy2

        for ny in range(height):
            for nx in range(width):
                usage[ny, nx] = 0.0

        if iteration % 2 == 0:
            start = 0
            end = self._n_query
            step = 1
            dx1, dy1 = -1, 0
            dx2, dy2 = 0, -1
        else:
            start = self._n_query - 1
            end = -1
            step = -1
            dx1, dy1 = 1, 0
            dx2, dy2 = 0, 1

        idx = start
        while idx != end:
            y = query_y[idx]
            x = query_x[idx]

            best_x = nnf[y, x, 0]
            best_y = nnf[y, x, 1]
            raw_dist = _patch_distance_fast(
                inpainted_features, y, x, source_features, best_y, best_x,
                half_patch_size, height, width, feat_channels
            )
            best_dist = raw_dist + usage_penalty * usage[best_y, best_x]

            ny = y + dy1
            nx = x + dx1
            if 0 <= ny < height and 0 <= nx < width:
                cand_x = nnf[ny, nx, 0] - dx1
                cand_y = nnf[ny, nx, 1] - dy1
                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_features, y, x, source_features, cand_y, cand_x,
                        half_patch_size, height, width, feat_channels
                    )
                    dist = dist + usage_penalty * usage[cand_y, cand_x]
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist
                        raw_dist = dist - usage_penalty * usage[cand_y, cand_x]

            ny = y + dy2
            nx = x + dx2
            if 0 <= ny < height and 0 <= nx < width:
                cand_x = nnf[ny, nx, 0] - dx2
                cand_y = nnf[ny, nx, 1] - dy2
                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_features, y, x, source_features, cand_y, cand_x,
                        half_patch_size, height, width, feat_channels
                    )
                    dist = dist + usage_penalty * usage[cand_y, cand_x]
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist
                        raw_dist = dist - usage_penalty * usage[cand_y, cand_x]

            search_radius = _max_int(height, width)
            while search_radius > 1:
                rx = _rand_int(&rng, 2 * search_radius + 1) - search_radius
                ry = _rand_int(&rng, 2 * search_radius + 1) - search_radius
                cand_x = best_x + rx
                cand_y = best_y + ry
                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_features, y, x, source_features, cand_y, cand_x,
                        half_patch_size, height, width, feat_channels
                    )
                    dist = dist + usage_penalty * usage[cand_y, cand_x]
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist
                        raw_dist = dist - usage_penalty * usage[cand_y, cand_x]
                search_radius //= 2

            # Store raw SSD (no usage) for voting similarity weights.
            raw_dist = _patch_distance_fast(
                inpainted_features, y, x, source_features, best_y, best_x,
                half_patch_size, height, width, feat_channels
            )
            nnf[y, x, 0] = best_x
            nnf[y, x, 1] = best_y
            nn_dist[y, x] = <DTYPE_t>raw_dist
            usage[best_y, best_x] += 1.0
            idx += step

        self._rng_state = rng

    cpdef void _reverse_search(self, np.ndarray[DTYPE_t, ndim=3] inpainted_features):
        cdef int i, s, q, sy, sx, qy, qx, best_qy, best_qx, old_mx, old_my
        cdef int n_samples, sample_idx, n_query, height, width, half_patch_size, feat_channels
        cdef int qdy, qdx
        cdef float best_d, current, dist
        cdef unsigned int rng = self._rng_state
        cdef int[:] ring_y = self._ring_y
        cdef int[:] ring_x = self._ring_x
        cdef int[:] query_y = self._query_y
        cdef int[:] query_x = self._query_x
        cdef np.uint8_t[:, :] query_mask = self.query_mask
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef DTYPE_t[:, :] nn_dist = self._nn_dist
        cdef DTYPE_t[:, :] usage = self._usage
        cdef DTYPE_t[:, :, :] features = inpainted_features
        cdef DTYPE_t[:, :, :] source_features = self.source_features
        cdef int q_offsets[4][2]

        if self._n_ring <= 0 or self._n_query <= 0:
            return

        height = self.height
        width = self.width
        half_patch_size = self.half_patch_size
        feat_channels = self.feat_channels
        n_query = self._n_query
        n_samples = n_query // 4
        if n_samples < 8:
            n_samples = 8
        if n_samples > 24:
            n_samples = 24

        q_offsets[0][0] = -1
        q_offsets[0][1] = 0
        q_offsets[1][0] = 1
        q_offsets[1][1] = 0
        q_offsets[2][0] = 0
        q_offsets[2][1] = -1
        q_offsets[3][0] = 0
        q_offsets[3][1] = 1

        for s in range(self._n_ring):
            sy = ring_y[s]
            sx = ring_x[s]
            best_qy = -1
            best_qx = -1
            best_d = 1e30

            for i in range(n_samples):
                sample_idx = _rand_int(&rng, n_query)
                qy = query_y[sample_idx]
                qx = query_x[sample_idx]
                dist = _patch_distance_fast(
                    features, qy, qx, source_features, sy, sx,
                    half_patch_size, height, width, feat_channels
                )
                if dist < best_d:
                    best_d = dist
                    best_qy = qy
                    best_qx = qx

            if best_qy >= 0:
                for i in range(4):
                    qdy = q_offsets[i][0]
                    qdx = q_offsets[i][1]
                    qy = best_qy + qdy
                    qx = best_qx + qdx
                    if qy < 0 or qy >= height or qx < 0 or qx >= width:
                        continue
                    if query_mask[qy, qx] == 0:
                        continue
                    dist = _patch_distance_fast(
                        features, qy, qx, source_features, sy, sx,
                        half_patch_size, height, width, feat_channels
                    )
                    if dist < best_d:
                        best_d = dist
                        best_qy = qy
                        best_qx = qx

            if best_qy < 0:
                continue

            current = nn_dist[best_qy, best_qx]
            if best_d >= current:
                continue

            old_mx = nnf[best_qy, best_qx, 0]
            old_my = nnf[best_qy, best_qx, 1]
            if usage[old_my, old_mx] > 0:
                usage[old_my, old_mx] -= 1.0
            nnf[best_qy, best_qx, 0] = sx
            nnf[best_qy, best_qx, 1] = sy
            nn_dist[best_qy, best_qx] = <DTYPE_t>best_d
            usage[sy, sx] += 1.0

        self._rng_state = rng

    def __call__(
        self,
        int iterations=5,
        bint randomize_nnf=True,
        np.ndarray[ITYPE_t, ndim=3] nnf=None,
        seed=None,
        int em_loops=1,
        bint reverse_search=False,
    ):
        cdef np.ndarray[DTYPE_t, ndim=3] inpainted_image = np.array(self.image).copy()
        cdef np.ndarray[DTYPE_t, ndim=3] inpainted_features
        cdef int it, em

        if em_loops < 1:
            raise ValueError("em_loops must be >= 1")

        if seed is not None:
            self._seed_rng(<unsigned int>int(seed))

        self._usage_penalty = 24.0 if reverse_search else 8.0

        if randomize_nnf:
            self._randomize_nnf()
        else:
            if nnf is None:
                raise ValueError("nnf must be provided when randomize_nnf is False")
            self._nnf = np.ascontiguousarray(nnf, dtype=np.int32)
            self._nn_dist = np.full((self.height, self.width), np.float32(1e20), dtype=np.float32)
            self._usage = np.zeros((self.height, self.width), dtype=np.float32)
            self._sanitize_nnf()

        inpainted_features = _build_feature_image_py(
            inpainted_image, edge_weight=self._edge_weight
        )
        self._initialize_distances(inpainted_features)
        self._paint_with_nnf(inpainted_image)
        inpainted_features = _build_feature_image_py(
            inpainted_image, edge_weight=self._edge_weight
        )

        for em in range(em_loops):
            for it in range(iterations):
                self._propagate_and_search(inpainted_features, it)
                self._paint_with_nnf(inpainted_image)
                inpainted_features = _build_feature_image_py(
                    inpainted_image, edge_weight=self._edge_weight
                )

            if reverse_search:
                self._reverse_search(inpainted_features)
                self._paint_with_nnf(inpainted_image)
                inpainted_features = _build_feature_image_py(
                    inpainted_image, edge_weight=self._edge_weight
                )

        return inpainted_image
