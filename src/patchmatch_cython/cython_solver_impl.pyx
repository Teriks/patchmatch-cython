# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand

# Type definitions
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t ITYPE_t

cdef inline int _max_int(int a, int b) nogil:
    return a if a > b else b

cdef inline int _min_int(int a, int b) nogil:
    return a if a < b else b

cdef inline float _patch_distance_fast(
    DTYPE_t[:, :, :] img1,
    int y1,
    int x1,
    DTYPE_t[:, :, :] img2,
    int y2,
    int x2,
    int half_patch_size,
    int height,
    int width) nogil:

    """Fast patch distance computation with bounds checking"""

    cdef int dy, dx, c
    cdef int py1, px1, py2, px2
    cdef float dist = 0.0
    cdef float diff
    cdef float val1, val2
    
    # Iterate over the full patch
    for dy in range(-half_patch_size, half_patch_size + 1):
        for dx in range(-half_patch_size, half_patch_size + 1):
            py1 = y1 + dy
            px1 = x1 + dx
            py2 = y2 + dy
            px2 = x2 + dx
            
            for c in range(3):
                # Check bounds and use actual pixel or 0
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
    """Cython implementation of PatchMatch algorithm"""
    
    cdef:
        DTYPE_t[:, :, :] image
        np.uint8_t[:, :] mask
        int patch_size, half_patch_size, height, width
        int[:] _masked_y, _masked_x
        int _n_masked
        np.uint8_t[:, :] _is_valid_coords
        int[:] _valid_y, _valid_x
        int _n_valid
        ITYPE_t[:, :, :] _nnf
        
    def __init__(
            self,
            np.ndarray[DTYPE_t, ndim=3] image,
            np.ndarray[np.uint8_t, ndim=2] mask,
            int patch_size
    ):

        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.half_patch_size = patch_size // 2
        self.height = image.shape[0]
        self.width = image.shape[1]
        
        # Get masked coordinates
        mask_indices = np.where(mask)
        self._masked_y = np.asarray(mask_indices[0], dtype=np.int32)
        self._masked_x = np.asarray(mask_indices[1], dtype=np.int32)
        self._n_masked = len(self._masked_y)
        
        # Create valid coordinates mask - use logical NOT to avoid bitwise issues
        self._is_valid_coords = np.logical_not(mask.astype(np.bool_)).astype(np.uint8)
        valid_indices = np.where(self._is_valid_coords)
        self._valid_y = np.asarray(valid_indices[0], dtype=np.int32)
        self._valid_x = np.asarray(valid_indices[1], dtype=np.int32)
        self._n_valid = len(self._valid_y)
        
        # Initialize NNF
        self._nnf = np.zeros((self.height, self.width, 2), dtype=np.int32)

    @property
    def nnf(self) -> np.ndarray:
        return np.asarray(self._nnf)

    @nnf.setter
    def nnf(self, value: np.ndarray):
        if value.dtype != np.int32:
            value = value.astype(np.int32, copy=False)  # copy only if needed

        # Ensure it's contiguous
        self._nnf = np.ascontiguousarray(value)

    cpdef void _randomize_nnf(self):
        """Initialize NNF with random valid coordinates"""
        cdef int i, y, x, rand_idx
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef int[:] valid_y = self._valid_y
        cdef int[:] valid_x = self._valid_x
        cdef int[:] masked_y = self._masked_y
        cdef int[:] masked_x = self._masked_x
        cdef np.uint8_t[:, :] mask = self.mask
        
        # Initialize with identity mapping for all pixels
        for y in range(self.height):
            for x in range(self.width):
                nnf[y, x, 0] = x
                nnf[y, x, 1] = y
        
        # Random assignment for masked pixels - only from valid pixels
        if self._n_valid > 0:
            for i in range(self._n_masked):
                y = masked_y[i]
                x = masked_x[i]
                # Select random valid pixel
                rand_idx = rand() % self._n_valid
                nnf[y, x, 0] = valid_x[rand_idx]
                nnf[y, x, 1] = valid_y[rand_idx]

    cpdef void _paint_with_nnf(self, np.ndarray[DTYPE_t, ndim=3] target):
        """Paint the target image using the NNF"""
        cdef int i, y, x, match_x, match_y, c
        cdef DTYPE_t[:, :, :] source = self.image
        cdef DTYPE_t[:, :, :] target_view = target
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef int[:] masked_y = self._masked_y
        cdef int[:] masked_x = self._masked_x
        
        for i in range(self._n_masked):
            y = masked_y[i]
            x = masked_x[i]
            match_x = nnf[y, x, 0]
            match_y = nnf[y, x, 1]
            for c in range(3):
                target_view[y, x, c] = source[match_y, match_x, c]
    
    cdef void _propagate_and_search(self, DTYPE_t[:, :, :] inpainted_image, int iteration):
        """Main propagation and search step"""
        cdef int idx, y, x, ny, nx, dy, dx
        cdef int best_x, best_y, cand_x, cand_y
        cdef float best_dist, dist
        cdef int search_radius, rx, ry
        cdef int height = self.height, width = self.width
        cdef int half_patch_size = self.half_patch_size
        cdef DTYPE_t[:, :, :] source = self.image
        cdef np.uint8_t[:, :] is_valid = self._is_valid_coords
        cdef ITYPE_t[:, :, :] nnf = self._nnf
        cdef int[:] masked_y = self._masked_y
        cdef int[:] masked_x = self._masked_x

        # Determine traversal direction
        cdef int start, end, step
        cdef int dx1, dy1, dx2, dy2

        if iteration % 2 == 0:
            start = 0
            end = self._n_masked
            step = 1
            dx1, dy1 = -1, 0  # left
            dx2, dy2 = 0, -1  # top
        else:
            start = self._n_masked - 1
            end = -1
            step = -1
            dx1, dy1 = 1, 0   # right
            dx2, dy2 = 0, 1   # bottom

        # Process each masked pixel
        idx = start
        while idx != end:
            y = masked_y[idx]
            x = masked_x[idx]

            # Current best match
            best_x = nnf[y, x, 0]
            best_y = nnf[y, x, 1]
            best_dist = _patch_distance_fast(
                inpainted_image, y, x, source, best_y, best_x, half_patch_size, height, width
            )

            # Propagation from neighbor 1
            ny = y + dy1
            nx = x + dx1
            if 0 <= ny < height and 0 <= nx < width:
                cand_x = nnf[ny, nx, 0] - dx1
                cand_y = nnf[ny, nx, 1] - dy1
                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_image, y, x, source, cand_y, cand_x, half_patch_size, height, width
                    )
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist

            # Propagation from neighbor 2
            ny = y + dy2
            nx = x + dx2
            if 0 <= ny < height and 0 <= nx < width:
                cand_x = nnf[ny, nx, 0] - dx2
                cand_y = nnf[ny, nx, 1] - dy2
                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_image, y, x, source, cand_y, cand_x, half_patch_size, height, width
                    )
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist

            # Random search
            search_radius = _max_int(height, width)
            while search_radius > 1:
                # Random offset
                rx = (rand() % (2 * search_radius + 1)) - search_radius
                ry = (rand() % (2 * search_radius + 1)) - search_radius

                cand_x = best_x + rx
                cand_y = best_y + ry

                if 0 <= cand_x < width and 0 <= cand_y < height and is_valid[cand_y, cand_x]:
                    dist = _patch_distance_fast(
                        inpainted_image, y, x, source, cand_y, cand_x, half_patch_size, height, width
                    )
                    if dist < best_dist:
                        best_x = cand_x
                        best_y = cand_y
                        best_dist = dist

                search_radius //= 2

            # Update NNF
            nnf[y, x, 0] = best_x
            nnf[y, x, 1] = best_y

            idx += step
    
    def __call__(
        self,
        int iterations=5,
        bint randomize_nnf=True,
        np.ndarray[ITYPE_t, ndim=3]
        nnf=None,
        seed=None
    ):
        
        """Run PatchMatch algorithm"""

        cdef np.ndarray[DTYPE_t, ndim=3] inpainted_image = np.array(self.image).copy()
        cdef int it
        
        # Re-seed if provided (allows run-time seed override)
        if seed is not None:
            srand(<unsigned int>seed)
        
        # Initialize NNF
        if randomize_nnf:
            self._randomize_nnf()
        else:
            self._nnf = nnf
        
        # Initial painting
        self._paint_with_nnf(inpainted_image)
        
        # Main iterations
        for it in range(iterations):
            self._propagate_and_search(inpainted_image, it)
            self._paint_with_nnf(inpainted_image)
        
        return inpainted_image
