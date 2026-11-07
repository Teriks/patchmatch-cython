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
import warnings

import numpy as np

from .solver import Solver

# Try to import Cython implementation
try:
    from patchmatch_cython.cython_solver_impl import CythonSolverImpl

    CYTHON_AVAILABLE: bool = True
except ImportError:
    CYTHON_AVAILABLE = False
    warnings.warn("Cython implementation not available. Run 'python setup.py build_ext --inplace' to build.")
    # Fall back to pure Python
    from .python_solver import PythonSolver as CythonSolverImpl


class CythonSolver(Solver):
    """
    Python wrapper for Cython-accelerated PatchMatch solver implementation.
    
    This class provides a high-performance implementation of the PatchMatch algorithm
    using Cython for acceleration. It automatically falls back to the pure Python
    implementation if the Cython extension is not available.
    
    The class wraps the low-level Cython implementation and handles type conversions
    between Python and Cython data types. It maintains the same interface as the
    base :class:`Solver` class while providing significant performance improvements.
    
    :param image: Input image with masked regions to be inpainted
    :type image: numpy.ndarray
    :param mask: Binary mask indicating regions to inpaint (True for masked pixels)
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches used for matching (must be odd)
    :type patch_size: int
    """

    def __init__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            patch_size: int,
            edge_weighted: bool = False,
    ) -> None:
        # Normalize mask to {0,1} uint8. Plain astype(uint8) leaves {0,255} masks
        # unchanged; boolean indexing bugs are avoided by going through != 0.
        image_f32 = np.ascontiguousarray(image, dtype=np.float32)
        mask_u8 = (np.asarray(mask) != 0).astype(np.uint8)
        super().__init__(
            image_f32, mask_u8.astype(bool), patch_size, edge_weighted=edge_weighted
        )

        self._solver: typing.Any = CythonSolverImpl(
            image_f32,
            np.ascontiguousarray(mask_u8),
            patch_size,
            edge_weighted=bool(edge_weighted),
        )

    @property
    def nnf(self) -> np.ndarray:
        return self._solver.nnf

    @nnf.setter
    def nnf(self, value: np.ndarray):
        self._solver.nnf = value

    def __call__(
            self,
            iterations: int = 5,
            randomize_nnf: bool = True,
            nnf: typing.Optional[np.ndarray] = None,
            seed: typing.Optional[int] = None,
            em_loops: int = 1,
            reverse_search: bool = False,
    ) -> np.ndarray:
        """
        Run the PatchMatch algorithm using Cython acceleration.
        
        Executes the Cython-accelerated PatchMatch algorithm to fill in masked regions
        of the source image. The method handles all necessary type conversions and
        delegates to the underlying Cython implementation for optimal performance.
        
        If the Cython extension is not available, this method will automatically
        fall back to the pure Python implementation with the same interface.
        
        :param iterations: Number of PatchMatch iterations to perform
        :type iterations: int
        :param randomize_nnf: Whether to randomly initialize the nearest neighbor field
        :type randomize_nnf: bool
        :param nnf: Pre-computed nearest neighbor field to use instead of
                   random initialization. Shape must be (height, width, 2)
        :type nnf: numpy.ndarray or None
        :param seed: Random seed for this specific run (overrides instance seed)
        :type seed: int or None
        :param em_loops: Outer match→vote→rematch loops (1 preserves prior behavior)
        :type em_loops: int
        :param reverse_search: Enable reverse/completeness search after each EM loop
        :type reverse_search: bool
        :return: Inpainted image with masked regions filled (float32)
        :rtype: numpy.ndarray
        """

        # Keep float32 here so callers (and inpaint_*) can round/saturate once.
        return np.asarray(
            self._solver(
                iterations,
                randomize_nnf,
                nnf,
                seed,
                em_loops,
                reverse_search,
            ),
            dtype=np.float32,
        )
