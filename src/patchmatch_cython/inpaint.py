import typing

import numpy as np

from .imageops import downsample, upsample
from .solver import Solver


def inpaint_pyramid(
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        solver_class: typing.Optional[typing.Type[Solver]] = None,
        seed: typing.Optional[int] = None
) -> np.ndarray:
    """
    Perform image inpainting using a multi-scale pyramid approach.
    
    This function implements a coarse-to-fine pyramid-based inpainting strategy
    that progressively refines the inpainting result across multiple resolution levels.
    The approach starts with a heavily downsampled version of the image and gradually
    works up to the full resolution, using the result from each level to initialize
    the next level.
    
    The pyramid approach significantly improves both the quality and speed of
    inpainting for large images by providing better initial estimates and reducing
    the search space at each level.
    
    :param image: Input image array with shape (H, W, C)
    :type image: numpy.ndarray
    :param mask: Binary mask indicating areas to inpaint with shape (H, W)
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches to use for matching
    :type patch_size: int
    :param solver_class: PatchMatch solver class to use. If None, defaults to
                        :class:`CythonSolver` with fallback to :class:`PythonSolver`
    :type solver_class: type or None
    :param seed: Random seed for reproducible results
    :type seed: int or None
    :return: Inpainted image as uint8 array with shape (H, W, C)
    :rtype: numpy.ndarray
    """
    if solver_class is None:
        # Default to Cython implementation (with fallback to Python)
        try:
            from .cython_solver import CythonSolver
            solver_class = CythonSolver
        except ImportError:
            from .python_solver import PythonSolver
            solver_class = PythonSolver

    image = image.copy().astype(np.float32)
    image[mask] = 0

    height, width = image.shape[:2]

    levels = int(np.floor(np.log2(min(height, width) / patch_size)))

    assert levels > 0

    levels_scales = 2 ** np.arange(levels + 1)

    height_list = height // levels_scales
    width_list = width // levels_scales

    nnf: typing.Optional[np.ndarray] = None

    for level in range(levels + 1):

        height = height_list[-level - 1]
        width = width_list[-level - 1]

        tmp_image: np.ndarray = downsample(image, height, width)
        tmp_mask: np.ndarray = downsample(mask.astype(np.float32), height, width).squeeze(-1).astype(bool)

        solver = solver_class(tmp_image, tmp_mask, patch_size)

        iterations: int = 2 * (level + 2) + 1

        if level == 0:
            inpainted_image: np.ndarray = solver(iterations=iterations, randomize_nnf=True, nnf=None, seed=seed)
        else:
            inpainted_image = solver(iterations=iterations, randomize_nnf=False, nnf=nnf, seed=seed)

        if level < levels:
            height = height_list[-level - 2]
            width = width_list[-level - 2]
            nnf = upsample(solver.nnf, height, width)

    # noinspection PyUnboundLocalVariable
    return inpainted_image.astype(np.uint8)


def inpaint_single(
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        iterations: int = 5,
        solver_class: typing.Optional[typing.Type[Solver]] = None,
        seed: typing.Optional[int] = None
) -> np.ndarray:
    """
    Perform image inpainting using a single resolution level.
    
    This function provides a straightforward single-scale inpainting approach
    that directly processes the image at its original resolution. This is simpler
    than the pyramid approach but may be slower for large images and potentially
    less effective for complex inpainting tasks.
    
    Use this function when you need fine control over the number of iterations
    or when working with small images where the pyramid approach is not necessary.
    
    :param image: Input image array with shape (H, W, C)
    :type image: numpy.ndarray
    :param mask: Binary mask indicating areas to inpaint with shape (H, W)
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches to use for matching
    :type patch_size: int
    :param iterations: Number of PatchMatch iterations to perform
    :type iterations: int
    :param solver_class: PatchMatch solver class to use. If None, defaults to
                        :class:`CythonSolver` with fallback to :class:`PythonSolver`
    :type solver_class: type or None
    :param seed: Random seed for reproducible results
    :type seed: int or None
    :return: Inpainted image as uint8 array with shape (H, W, C)
    :rtype: numpy.ndarray
    """
    if solver_class is None:
        # Default to Cython implementation (with fallback to Python)
        try:
            from .cython_solver import CythonSolver
            solver_class = CythonSolver
        except ImportError:
            from .python_solver import PythonSolver
            solver_class = PythonSolver

    # noinspection PyUnboundLocalVariable
    return solver_class(image, mask, patch_size)(iterations=iterations, seed=seed).astype(np.uint8)
