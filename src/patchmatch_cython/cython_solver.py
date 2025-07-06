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
            patch_size: int
    ) -> None:
        super().__init__(image, mask, patch_size)

        self._solver: typing.Any = CythonSolverImpl(
            image.astype(np.float32),
            mask.astype(np.uint8),
            patch_size
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
            seed: typing.Optional[int] = None
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
        :return: Inpainted image with masked regions filled, converted to uint8
        :rtype: numpy.ndarray
        """

        return self._solver(iterations, randomize_nnf, nnf, seed).astype(np.uint8)
