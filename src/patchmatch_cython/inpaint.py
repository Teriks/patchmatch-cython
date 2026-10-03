import typing

import numpy as np

from .imageops import (
    build_query_patch_centers,
    build_valid_patch_centers,
    downsample,
    feather_compose,
    poisson_blend as _poisson_blend,
    prepare_inpaint_inputs,
    resize_image,
    sanitize_nnf,
    to_uint8_image,
    upsample,
)
from .solver import Solver


def _default_solver_class() -> typing.Type[Solver]:
    try:
        from .cython_solver import CythonSolver
        return CythonSolver
    except ImportError:
        from .python_solver import PythonSolver
        return PythonSolver


def _validate_quality_options(
        em_loops: int,
        hole_refine: int,
        mask_feather: float,
) -> None:
    if em_loops < 1:
        raise ValueError("em_loops must be >= 1")
    if hole_refine < 0:
        raise ValueError("hole_refine must be >= 0")
    if mask_feather < 0:
        raise ValueError("mask_feather must be >= 0")


def _compose_result(
        original: np.ndarray,
        mask_bool: np.ndarray,
        filled: np.ndarray,
        poisson_blend: bool,
        mask_feather: float,
) -> np.ndarray:
    composed = original.copy()
    composed[mask_bool] = filled[mask_bool]

    if mask_feather > 0:
        # Feather handles its own harmonic rim blend at the given width.
        composed = feather_compose(original, mask_bool, filled, feather=mask_feather)
    elif poisson_blend:
        composed = _poisson_blend(original, mask_bool, composed)

    return to_uint8_image(composed)


def _hole_refine_pass(
        original: np.ndarray,
        mask_bool: np.ndarray,
        filled: np.ndarray,
        nnf: typing.Optional[np.ndarray],
        patch_size: int,
        solver_class: typing.Type[Solver],
        hole_refine: int,
        edge_weighted: bool,
        reverse_search: bool,
        seed: typing.Optional[int],
) -> np.ndarray:
    """Extra full-resolution PatchMatch iterations on the already-filled hole."""
    if hole_refine <= 0:
        return filled

    warm = original.copy()
    warm[mask_bool] = filled[mask_bool]
    solver = solver_class(
        warm, mask_bool, patch_size, edge_weighted=edge_weighted
    )

    if nnf is not None:
        half_patch = patch_size // 2
        nnf_safe = sanitize_nnf(
            nnf,
            build_query_patch_centers(mask_bool, half_patch),
            build_valid_patch_centers(mask_bool, half_patch),
            rng=np.random.default_rng(seed) if seed is not None else None,
        )
        return solver(
            iterations=hole_refine,
            randomize_nnf=False,
            nnf=nnf_safe,
            seed=seed,
            em_loops=1,
            reverse_search=reverse_search,
        )

    return solver(
        iterations=hole_refine,
        randomize_nnf=True,
        nnf=None,
        seed=seed,
        em_loops=1,
        reverse_search=reverse_search,
    )


def inpaint_pyramid(
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        solver_class: typing.Optional[typing.Type[Solver]] = None,
        seed: typing.Optional[int] = None,
        em_loops: int = 1,
        edge_weighted: bool = False,
        reverse_search: bool = False,
        poisson_blend: bool = True,
        hole_refine: int = 0,
        mask_feather: float = 0.0,
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
    :param mask: Binary mask indicating areas to inpaint with shape (H, W).
                 Accepts bool, ``{0,1}``, or ``{0,255}`` masks.
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches to use for matching
    :type patch_size: int
    :param solver_class: PatchMatch solver class to use. If None, defaults to
                        :class:`CythonSolver` with fallback to :class:`PythonSolver`
    :type solver_class: type or None
    :param seed: Random seed for reproducible results
    :type seed: int or None
    :param em_loops: Outer match→vote→rematch loops at coarse pyramid levels
                     (finest level always uses 1). Default ``1`` keeps prior behavior.
    :type em_loops: int
    :param edge_weighted: Emphasize edge/structure channels in patch distance
    :type edge_weighted: bool
    :param reverse_search: Enable reverse/completeness search after each EM loop
    :type reverse_search: bool
    :param poisson_blend: Soften hole seams with a harmonic/guide cross-fade
                          (default band 8px). Ignored when ``mask_feather > 0``,
                          which supplies its own seam width.
    :type poisson_blend: bool
    :param hole_refine: Extra full-resolution PatchMatch iterations after the
                        pyramid (0 disables). Operates only via hole painting.
    :type hole_refine: int
    :param mask_feather: Soft compose width in pixels from the hole boundary
                         (0 = hard paste / optional poisson only). Does not
                         alter the mask used for matching.
    :type mask_feather: float
    :return: Inpainted image as uint8 array with shape (H, W, C)
    :rtype: numpy.ndarray
    """
    _validate_quality_options(em_loops, hole_refine, mask_feather)

    if solver_class is None:
        solver_class = _default_solver_class()

    original, working, mask_bool = prepare_inpaint_inputs(image, mask, patch_size)
    if not np.any(mask_bool):
        return to_uint8_image(original)

    height, width = working.shape[:2]
    levels = int(np.floor(np.log2(min(height, width) / patch_size)))
    if levels <= 0:
        # Image too small for a pyramid; fall back to single-scale.
        return inpaint_single(
            original,
            mask_bool,
            patch_size=patch_size,
            iterations=7,
            solver_class=solver_class,
            seed=seed,
            em_loops=em_loops,
            edge_weighted=edge_weighted,
            reverse_search=reverse_search,
            poisson_blend=poisson_blend,
            hole_refine=hole_refine,
            mask_feather=mask_feather,
        )

    levels_scales = 2 ** np.arange(levels + 1)
    height_list = height // levels_scales
    width_list = width // levels_scales

    nnf: typing.Optional[np.ndarray] = None
    prev_fill: typing.Optional[np.ndarray] = None
    mask_float = mask_bool.astype(np.float32)
    inpainted_image = working
    solver: typing.Optional[Solver] = None

    for level in range(levels + 1):
        level_h = int(height_list[-level - 1])
        level_w = int(width_list[-level - 1])

        tmp_image = downsample(working, level_h, level_w)
        tmp_mask = downsample(mask_float, level_h, level_w).squeeze(-1) > 0

        # Warm-start hole content from the upsampled coarser fill.
        if prev_fill is not None:
            up_fill = resize_image(prev_fill, level_h, level_w)
            tmp_image = tmp_image.copy()
            tmp_image[tmp_mask] = up_fill[tmp_mask]

        solver = solver_class(
            tmp_image, tmp_mask, patch_size, edge_weighted=edge_weighted
        )
        iterations = 2 * (level + 2) + 1
        # EM loops are most useful (and cheapest) at coarse levels.
        level_em_loops = em_loops if level < levels else 1

        if level == 0:
            inpainted_image = solver(
                iterations=iterations,
                randomize_nnf=True,
                nnf=None,
                seed=seed,
                em_loops=level_em_loops,
                reverse_search=reverse_search,
            )
        else:
            inpainted_image = solver(
                iterations=iterations,
                randomize_nnf=False,
                nnf=nnf,
                seed=seed,
                em_loops=level_em_loops,
                reverse_search=reverse_search,
            )

        prev_fill = inpainted_image

        if level < levels:
            next_height = int(height_list[-level - 2])
            next_width = int(width_list[-level - 2])
            next_mask = downsample(mask_float, next_height, next_width).squeeze(-1) > 0
            half_patch = patch_size // 2
            valid_centers = build_valid_patch_centers(next_mask, half_patch)
            query_centers = build_query_patch_centers(next_mask, half_patch)
            rng = np.random.default_rng(seed) if seed is not None else None
            nnf = sanitize_nnf(
                upsample(solver.nnf, next_height, next_width),
                query_centers,
                valid_centers,
                rng=rng,
            )

    assert solver is not None
    inpainted_image = _hole_refine_pass(
        original,
        mask_bool,
        inpainted_image,
        solver.nnf,
        patch_size,
        solver_class,
        hole_refine,
        edge_weighted,
        reverse_search,
        seed,
    )

    return _compose_result(
        original, mask_bool, inpainted_image, poisson_blend, mask_feather
    )


def inpaint_single(
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        iterations: int = 5,
        solver_class: typing.Optional[typing.Type[Solver]] = None,
        seed: typing.Optional[int] = None,
        em_loops: int = 1,
        edge_weighted: bool = False,
        reverse_search: bool = False,
        poisson_blend: bool = True,
        hole_refine: int = 0,
        mask_feather: float = 0.0,
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
    :param mask: Binary mask indicating areas to inpaint with shape (H, W).
                 Accepts bool, ``{0,1}``, or ``{0,255}`` masks.
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
    :param em_loops: Outer match→vote→rematch loops. Default ``1`` keeps prior behavior.
    :type em_loops: int
    :param edge_weighted: Emphasize edge/structure channels in patch distance
    :type edge_weighted: bool
    :param reverse_search: Enable reverse/completeness search after each EM loop
    :type reverse_search: bool
    :param poisson_blend: Soften hole seams with a harmonic/guide cross-fade
    :type poisson_blend: bool
    :param hole_refine: Extra PatchMatch iterations after the main solve (0 disables)
    :type hole_refine: int
    :param mask_feather: Soft compose width in pixels from the hole boundary
    :type mask_feather: float
    :return: Inpainted image as uint8 array with shape (H, W, C)
    :rtype: numpy.ndarray
    """
    _validate_quality_options(em_loops, hole_refine, mask_feather)

    if solver_class is None:
        solver_class = _default_solver_class()

    original, working, mask_bool = prepare_inpaint_inputs(image, mask, patch_size)
    if not np.any(mask_bool):
        return to_uint8_image(original)

    solver = solver_class(
        working, mask_bool, patch_size, edge_weighted=edge_weighted
    )
    result = solver(
        iterations=iterations,
        seed=seed,
        em_loops=em_loops,
        reverse_search=reverse_search,
    )
    result = _hole_refine_pass(
        original,
        mask_bool,
        result,
        solver.nnf,
        patch_size,
        solver_class,
        hole_refine,
        edge_weighted,
        reverse_search,
        seed,
    )
    return _compose_result(
        original, mask_bool, result, poisson_blend, mask_feather
    )
