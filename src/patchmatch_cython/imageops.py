import numpy as np
from typing import Union

# Try to import OpenCV for optimized operations. Keep the module handle even when
# HAS_OPENCV is forced False so benchmarks can toggle acceleration at runtime.
try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

HAS_OPENCV = _cv2 is not None


def set_use_opencv(enabled: bool) -> bool:
    """
    Enable/disable OpenCV-accelerated imageops paths at runtime.

    Returns the effective enabled state (always False if OpenCV is not installed).
    """
    global HAS_OPENCV
    HAS_OPENCV = bool(enabled) and _cv2 is not None
    return HAS_OPENCV


def opencv_available() -> bool:
    """Return True if OpenCV is importable (regardless of the runtime toggle)."""
    return _cv2 is not None


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a mask to a boolean array.

    Accepts bool, ``{0,1}``, or ``{0,255}`` masks. Integer/float masks must not be
    used directly for NumPy indexing — non-bool dtypes are treated as fancy
    integer indexes, not as boolean masks.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.dtype == np.bool_:
        return mask_arr
    return mask_arr != 0


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Round and saturate a floating-point image to ``uint8``."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def prepare_inpaint_inputs(
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and normalize ``(image, mask, patch_size)`` for inpainting.

    Returns ``(original_float32, working_float32, mask_bool)`` where ``working``
    has hole pixels zeroed and ``original`` keeps source pixels for compositing.
    """
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")

    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        image_arr = image_arr[:, :, None]
    if image_arr.ndim != 3:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if image_arr.shape[0] < patch_size or image_arr.shape[1] < patch_size:
        raise ValueError(
            f"image spatial size {image_arr.shape[:2]} is smaller than patch_size={patch_size}"
        )

    mask_bool = as_bool_mask(mask)
    if mask_bool.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if mask_bool.shape != image_arr.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} does not match image shape {image_arr.shape[:2]}"
        )

    original = np.array(image_arr, dtype=np.float32, copy=True)
    if not np.any(mask_bool):
        return original, original.copy(), mask_bool
    if np.all(mask_bool):
        raise ValueError("mask covers the entire image; no source pixels available")

    working = original.copy()
    working[mask_bool] = 0
    return original, working, mask_bool


def resize_image(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Bilinear-resize an image to ``(new_height, new_width)`` as float32."""
    image_f = np.ascontiguousarray(image, dtype=np.float32)
    if image_f.shape[0] == new_height and image_f.shape[1] == new_width:
        return image_f.copy()

    if HAS_OPENCV:
        resized = _cv2.resize(
            image_f, (new_width, new_height), interpolation=_cv2.INTER_LINEAR
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return resized.astype(np.float32, copy=False)

    h, w = image_f.shape[:2]
    channels = 1 if image_f.ndim == 2 else image_f.shape[2]
    if image_f.ndim == 2:
        image_f = image_f[:, :, None]

    y_out, x_out = np.mgrid[0:new_height, 0:new_width]
    y_in = (y_out + 0.5) * h / new_height - 0.5
    x_in = (x_out + 0.5) * w / new_width - 0.5
    x0 = np.floor(x_in).astype(int)
    y0 = np.floor(y_in).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x_in - x0
    wy = y_in - y0
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    result = np.zeros((new_height, new_width, channels), dtype=np.float32)
    for c in range(channels):
        result[..., c] = (
            image_f[y0, x0, c] * (1 - wx) * (1 - wy)
            + image_f[y0, x1, c] * wx * (1 - wy)
            + image_f[y1, x0, c] * (1 - wx) * wy
            + image_f[y1, x1, c] * wx * wy
        )
    return result


def hole_distance_to_known(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance from each hole pixel to the nearest known pixel.

    Known pixels (and empty masks) are 0. Used for seam blending / feathering.
    """
    mask_bool = as_bool_mask(mask)
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.float32)

    if HAS_OPENCV:
        # Non-zero pixels get distance to nearest zero pixel.
        return _cv2.distanceTransform(
            mask_bool.astype(np.uint8), _cv2.DIST_L2, 5
        ).astype(np.float32)

    # City-block distance via layered peeling (no SciPy required).
    dist = np.zeros(mask_bool.shape, dtype=np.float32)
    remaining = mask_bool.copy()
    current_d = 0.0
    while np.any(remaining):
        current_d += 1.0
        outside = ~remaining
        neigh = np.zeros_like(remaining)
        neigh[1:, :] |= outside[:-1, :]
        neigh[:-1, :] |= outside[1:, :]
        neigh[:, 1:] |= outside[:, :-1]
        neigh[:, :-1] |= outside[:, 1:]
        layer = remaining & neigh
        if not np.any(layer):
            dist[remaining] = current_d
            break
        dist[layer] = current_d
        remaining[layer] = False
    return dist


def _harmonic_fill(
        source: np.ndarray,
        mask_bool: np.ndarray,
        guide: np.ndarray,
        iterations: int,
) -> np.ndarray:
    """Jacobi membrane fill: ``∇²u = 0`` in the hole, ``u = source`` on known."""
    src = np.ascontiguousarray(source, dtype=np.float32)
    g = np.ascontiguousarray(guide, dtype=np.float32)
    if src.ndim == 2:
        src = src[:, :, None]
        g = g[:, :, None]

    hole_y, hole_x = np.where(mask_bool)
    height, width = mask_bool.shape
    y0 = max(int(hole_y.min()) - 1, 0)
    y1 = min(int(hole_y.max()) + 2, height)
    x0 = max(int(hole_x.min()) - 1, 0)
    x1 = min(int(hole_x.max()) + 2, width)

    src_c = src[y0:y1, x0:x1]
    g_c = g[y0:y1, x0:x1]
    mask_c = mask_bool[y0:y1, x0:x1]
    known_c = ~mask_c

    u = g_c.copy()
    u[known_c] = src_c[known_c]

    for _ in range(iterations):
        up = np.empty_like(u)
        down = np.empty_like(u)
        left = np.empty_like(u)
        right = np.empty_like(u)
        up[0] = u[0]
        up[1:] = u[:-1]
        down[-1] = u[-1]
        down[:-1] = u[1:]
        left[:, 0] = u[:, 0]
        left[:, 1:] = u[:, :-1]
        right[:, -1] = u[:, -1]
        right[:, :-1] = u[:, 1:]
        u_new = 0.25 * (up + down + left + right)
        u_new[known_c] = src_c[known_c]
        u = u_new

    out = src.copy()
    out[y0:y1, x0:x1][mask_c] = u[mask_c]
    return out


def poisson_blend(
        source: np.ndarray,
        mask: np.ndarray,
        guide: np.ndarray,
        iterations: int | None = None,
        blend_band: float = 8.0,
) -> np.ndarray:
    """
    Soften hole seams by blending ``guide`` with a harmonic fill near the boundary.

    Plain gradient-domain cloning is a no-op when ``guide`` already matches
    ``source`` on known pixels (the inpaint compose path). Instead this solves a
    membrane (``∇²u = 0``) with Dirichlet boundaries from ``source``, then
    cross-fades to ``guide`` with distance from the hole edge so interior
    PatchMatch texture is preserved while the rim is seam-smoothed.

    :param blend_band: Distance (px) over which the harmonic→guide fade occurs.
    """
    mask_bool = as_bool_mask(mask)
    if not np.any(mask_bool):
        return np.asarray(source, dtype=np.float32)

    src = np.ascontiguousarray(source, dtype=np.float32)
    g = np.ascontiguousarray(guide, dtype=np.float32)
    if src.ndim == 2:
        src = src[:, :, None]
        g = g[:, :, None]
    if blend_band <= 0:
        out = src.copy()
        out[mask_bool] = g[mask_bool]
        return out

    hole_count = int(np.count_nonzero(mask_bool))
    if iterations is None:
        iterations = int(np.clip(np.sqrt(hole_count), 24, 80))

    harmonic = _harmonic_fill(src, mask_bool, g, iterations=iterations)
    dist = hole_distance_to_known(mask_bool)
    alpha = np.clip(dist / np.float32(blend_band), 0.0, 1.0).astype(np.float32)

    out = src.copy()
    a = alpha[mask_bool][:, None]
    out[mask_bool] = a * g[mask_bool] + (1.0 - a) * harmonic[mask_bool]
    return out


def feather_compose(
        source: np.ndarray,
        mask: np.ndarray,
        filled: np.ndarray,
        feather: float,
) -> np.ndarray:
    """
    Soft-compose ``filled`` into ``source`` over ``mask`` using a distance ramp.

    Does not modify the mask used for matching. ``feather`` is the hole-boundary
    width in pixels: rim pixels blend toward a harmonic boundary pull (not the
    original hole content), and the deep interior keeps ``filled``.
    """
    if feather <= 0:
        out = np.ascontiguousarray(source, dtype=np.float32).copy()
        if out.ndim == 2:
            out = out[:, :, None]
        filled_f = np.ascontiguousarray(filled, dtype=np.float32)
        if filled_f.ndim == 2:
            filled_f = filled_f[:, :, None]
        out[as_bool_mask(mask)] = filled_f[as_bool_mask(mask)]
        return out

    mask_bool = as_bool_mask(mask)
    src = np.ascontiguousarray(source, dtype=np.float32)
    filled_f = np.ascontiguousarray(filled, dtype=np.float32)
    if src.ndim == 2:
        src = src[:, :, None]
        filled_f = filled_f[:, :, None]

    guide = src.copy()
    guide[mask_bool] = filled_f[mask_bool]
    # Lightweight rim reference (few Jacobi iters); fade width is ``feather``.
    return poisson_blend(
        src,
        mask_bool,
        guide,
        iterations=max(12, int(feather * 3)),
        blend_band=float(feather),
    )


def _window_sum(binary: np.ndarray, half_patch_size: int) -> np.ndarray:
    """Integral-image sum of a binary map over each patch-sized window."""
    patch_span = 2 * half_patch_size + 1
    values = binary.astype(np.int32, copy=False)
    padded = np.pad(values, half_patch_size, mode="constant", constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    height, width = binary.shape
    y0 = np.arange(height)[:, None]
    x0 = np.arange(width)[None, :]
    return (
        integral[y0 + patch_span, x0 + patch_span]
        - integral[y0, x0 + patch_span]
        - integral[y0 + patch_span, x0]
        + integral[y0, x0]
    )


def _morph_kernel(half_patch_size: int) -> np.ndarray:
    patch_span = 2 * half_patch_size + 1
    return np.ones((patch_span, patch_span), dtype=np.uint8)


def build_valid_patch_centers(mask: np.ndarray, half_patch_size: int) -> np.ndarray:
    """
    Return a boolean map of source patch centers whose full neighborhood is known.

    A center is valid only if every pixel in its ``(2 * half_patch_size + 1)`` square
    is in-bounds and unmasked. If that set is empty (very large holes), fall back to
    all unmasked pixels so matching can still proceed.
    """
    mask_bool = as_bool_mask(mask)
    known = np.logical_not(mask_bool)
    if half_patch_size <= 0:
        return known

    if HAS_OPENCV:
        # Erosion == "every pixel in the patch neighborhood is known".
        known_u8 = known.astype(np.uint8)
        eroded = _cv2.erode(
            known_u8,
            _morph_kernel(half_patch_size),
            borderType=_cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = eroded.astype(bool)
    else:
        patch_span = 2 * half_patch_size + 1
        valid = _window_sum(known, half_patch_size) == (patch_span * patch_span)

    if not np.any(valid):
        return known
    return valid


def build_query_patch_centers(mask: np.ndarray, half_patch_size: int) -> np.ndarray:
    """
    Return centers whose patch overlaps the hole (hole pixels plus a border band).

    Matching these centers improves boundary coherence versus hole-only PatchMatch.
    """
    mask_bool = as_bool_mask(mask)
    if half_patch_size <= 0:
        return mask_bool

    if HAS_OPENCV:
        # Dilation == "any pixel in the patch neighborhood is masked".
        dilated = _cv2.dilate(
            mask_bool.astype(np.uint8),
            _morph_kernel(half_patch_size),
            borderType=_cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return dilated.astype(bool)

    return _window_sum(mask_bool, half_patch_size) > 0


def build_spatial_vote_weights(half_patch_size: int) -> np.ndarray:
    """Gaussian spatial weights for patch voting, shape ``(2h+1, 2h+1)`` float32."""
    if half_patch_size <= 0:
        return np.ones((1, 1), dtype=np.float32)
    sigma = max(half_patch_size / 2.0, 0.5)
    offsets = np.arange(-half_patch_size, half_patch_size + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return weights.astype(np.float32)


def build_source_ring_centers(mask: np.ndarray, half_patch_size: int) -> np.ndarray:
    """
    Valid source centers near the hole for reverse / completeness search.

    These are fully-known patch centers that lie in a band around the hole, so
    reverse search can push underused local texture into the NNF.
    """
    mask_bool = as_bool_mask(mask)
    valid = build_valid_patch_centers(mask_bool, half_patch_size)
    ring_radius = max(half_patch_size * 2, half_patch_size + 1)
    near_hole = build_query_patch_centers(mask_bool, ring_radius)
    return near_hole & valid


def build_feature_image(
        image: np.ndarray,
        edge_weight: float = 1.0,
        mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build matching features from color, optionally with gradient channels.

    Gradients only affect distance, not painting. Works for grayscale
    ``(H,W,1)`` and multi-channel images.

    :param edge_weight: Multiplier for gradient channels. ``0`` means color-only
        (no gradient channels). Values ``> 0`` append ``gx, gy``.
    :param mask: Optional hole mask. When set, gradient samples that touch the
        hole are zeroed so zeroed-hole edges do not create fake structure.
    """
    data = np.ascontiguousarray(image, dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if edge_weight < 0:
        raise ValueError("edge_weight must be >= 0")

    if data.shape[2] >= 3:
        gray = (
            0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
        ).astype(np.float32, copy=False)
        color = data[..., :3]
    else:
        gray = data[..., 0].astype(np.float32, copy=False)
        color = data

    if edge_weight == 0:
        return np.ascontiguousarray(color, dtype=np.float32)

    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    grad_y[1:-1, :] = gray[2:, :] - gray[:-2, :]

    if mask is not None:
        mask_bool = as_bool_mask(mask)
        # Zero gradients whose finite-difference stencil touches the hole.
        invalid_x = np.zeros(mask_bool.shape, dtype=bool)
        invalid_y = np.zeros(mask_bool.shape, dtype=bool)
        invalid_x[:, 1:-1] = mask_bool[:, 2:] | mask_bool[:, :-2] | mask_bool[:, 1:-1]
        invalid_y[1:-1, :] = mask_bool[2:, :] | mask_bool[:-2, :] | mask_bool[1:-1, :]
        grad_x[invalid_x] = 0
        grad_y[invalid_y] = 0

    if edge_weight != 1.0:
        grad_x *= np.float32(edge_weight)
        grad_y *= np.float32(edge_weight)
    return np.concatenate([color, grad_x[..., None], grad_y[..., None]], axis=2)


def sanitize_nnf(
        nnf: np.ndarray,
        mask: np.ndarray,
        valid_centers: np.ndarray,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Clamp NNF coordinates and replace active entries that point outside valid centers.
    """
    nnf = np.ascontiguousarray(nnf, dtype=np.int32)
    active = as_bool_mask(mask)
    height, width = active.shape
    nnf[..., 0] = np.clip(nnf[..., 0], 0, width - 1)
    nnf[..., 1] = np.clip(nnf[..., 1], 0, height - 1)

    active_y, active_x = np.where(active)
    if active_y.size == 0:
        return nnf

    match_x = nnf[active_y, active_x, 0]
    match_y = nnf[active_y, active_x, 1]
    invalid = ~valid_centers[match_y, match_x]
    if not np.any(invalid):
        return nnf

    valid_y, valid_x = np.where(valid_centers)
    if valid_y.size == 0:
        return nnf

    if rng is None:
        rng = np.random.default_rng()

    n_invalid = int(np.count_nonzero(invalid))
    choices = rng.integers(0, valid_y.size, size=n_invalid)
    nnf[active_y[invalid], active_x[invalid], 0] = valid_x[choices]
    nnf[active_y[invalid], active_x[invalid], 1] = valid_y[choices]
    return nnf


def downsample(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Downsample an image to the specified dimensions using mean/area pooling.
    
    Crops to an integer number of blocks first (same as the NumPy path), then
    averages. With OpenCV this uses ``INTER_AREA``; otherwise a reshape/mean
    implementation is used.
    
    :param image: Input image array with shape (H, W) or (H, W, C)
    :type image: numpy.ndarray
    :param new_height: Target height for the downsampled image
    :type new_height: int
    :param new_width: Target width for the downsampled image
    :type new_width: int
    :return: Downsampled image as float32 array with shape (new_height, new_width, C)
    :rtype: numpy.ndarray
    """
    h, w = image.shape[:2]
    cropped = np.ascontiguousarray(
        image[: new_height * (h // new_height), : new_width * (w // new_width)],
        dtype=np.float32,
    )
    if cropped.ndim == 2:
        cropped = cropped[:, :, None]

    if HAS_OPENCV:
        out = _cv2.resize(
            cropped, (new_width, new_height), interpolation=_cv2.INTER_AREA
        )
        if out.ndim == 2:
            out = out[:, :, None]
        return out.astype(np.float32, copy=False)

    cropped = cropped.reshape(
        new_height, h // new_height, new_width, w // new_width, -1
    )
    return cropped.mean(axis=(1, 3)).astype(np.float32)


def bilinear_interpolate(array: np.ndarray, y: float, x: float) -> Union[float, np.ndarray]:
    """
    Perform bilinear interpolation at a specific point in an array.
    
    This function computes the interpolated value at the given floating-point
    coordinates using bilinear interpolation. It handles boundary conditions
    gracefully by clamping coordinates to the valid range.
    
    :param array: Input array to interpolate from with shape (H, W) or (H, W, C)
    :type array: numpy.ndarray
    :param y: Y coordinate (row) for interpolation
    :type y: float
    :param x: X coordinate (column) for interpolation
    :type x: float
    :return: Interpolated value. Returns scalar for 2D arrays, array for 3D arrays
    :rtype: float or numpy.ndarray
    """
    h, w = array.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    
    # Compute weights
    wx: float = x - x0
    wy: float = y - y0
    
    # Bilinear interpolation
    if len(array.shape) == 3:
        result: np.ndarray = (1 - wy) * (1 - wx) * array[y0, x0] + \
                (1 - wy) * wx * array[y0, x1] + \
                wy * (1 - wx) * array[y1, x0] + \
                wy * wx * array[y1, x1]
    else:
        result = (1 - wy) * (1 - wx) * array[y0, x0] + \
                (1 - wy) * wx * array[y0, x1] + \
                wy * (1 - wx) * array[y1, x0] + \
                wy * wx * array[y1, x1]
    
    return result


def _upsample_opencv(nnf: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Upsample a nearest neighbor field using OpenCV's optimized bilinear interpolation.
    
    This function uses OpenCV's highly optimized ``cv2.resize`` function to
    upsample the nearest neighbor field. It includes proper coordinate scaling
    to ensure the upsampled coordinates map correctly to the new dimensions.
    
    :param nnf: Nearest neighbor field to upsample with shape (H, W, 2)
    :type nnf: numpy.ndarray
    :param new_height: Target height for the upsampled NNF
    :type new_height: int
    :param new_width: Target width for the upsampled NNF
    :type new_width: int
    :return: Upsampled NNF as int32 array with shape (new_height, new_width, 2)
    :rtype: numpy.ndarray
    """
    h, w = nnf.shape[:2]
    nnf = nnf.astype(np.float32)
    
    # Scale coordinates
    nnf[..., 0] = (nnf[..., 0] + 0.5) * new_width / w - 0.5
    nnf[..., 1] = (nnf[..., 1] + 0.5) * new_height / h - 0.5
    
    # Use OpenCV's optimized bilinear interpolation
    nnf = _cv2.resize(nnf, (new_width, new_height), interpolation=_cv2.INTER_LINEAR)
    nnf = np.round(nnf).astype(np.int32)
    nnf[..., 0] = np.clip(nnf[..., 0], 0, new_width - 1)
    nnf[..., 1] = np.clip(nnf[..., 1], 0, new_height - 1)
    return nnf


def _upsample_numpy(nnf: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Upsample a nearest neighbor field using optimized pure NumPy implementation.
    
    This function provides a pure NumPy implementation of bilinear interpolation
    that matches OpenCV's quality and behavior. It uses vectorized operations
    for optimal performance when OpenCV is not available.
    
    The implementation handles proper coordinate scaling and boundary conditions
    to ensure high-quality upsampling results.
    
    :param nnf: Nearest neighbor field to upsample with shape (H, W, 2)
    :type nnf: numpy.ndarray
    :param new_height: Target height for the upsampled NNF
    :type new_height: int
    :param new_width: Target width for the upsampled NNF
    :type new_width: int
    :return: Upsampled NNF as int32 array with shape (new_height, new_width, 2)
    :rtype: numpy.ndarray
    """
    h, w = nnf.shape[:2]
    nnf = nnf.astype(np.float32)
    
    # Scale coordinates first (same as OpenCV)
    nnf[..., 0] = (nnf[..., 0] + 0.5) * new_width / w - 0.5
    nnf[..., 1] = (nnf[..., 1] + 0.5) * new_height / h - 0.5
    
    # Create output coordinate grids
    y_out, x_out = np.mgrid[0:new_height, 0:new_width]
    
    # Map output coordinates to input coordinates
    y_in = (y_out + 0.5) * h / new_height - 0.5
    x_in = (x_out + 0.5) * w / new_width - 0.5
    
    # Find integer coordinates and fractional parts
    x0 = np.floor(x_in).astype(int)
    y0 = np.floor(y_in).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Fractional parts
    wx = x_in - x0
    wy = y_in - y0
    
    # Clip coordinates to image bounds
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # Perform bilinear interpolation for both channels
    result = np.zeros((new_height, new_width, 2), dtype=np.float32)
    
    for c in range(2):  # For x and y coordinates
        # Get the four corner values
        val_00 = nnf[y0, x0, c]
        val_01 = nnf[y0, x1, c]
        val_10 = nnf[y1, x0, c]
        val_11 = nnf[y1, x1, c]
        
        # Bilinear interpolation
        result[..., c] = (val_00 * (1 - wx) * (1 - wy) +
                         val_01 * wx * (1 - wy) +
                         val_10 * (1 - wx) * wy +
                         val_11 * wx * wy)

    nnf = np.round(result).astype(np.int32)
    nnf[..., 0] = np.clip(nnf[..., 0], 0, new_width - 1)
    nnf[..., 1] = np.clip(nnf[..., 1], 0, new_height - 1)
    return nnf


def upsample(nnf: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Upsample a nearest neighbor field to the specified dimensions using bilinear interpolation.
    
    This function provides an adaptive upsampling approach that uses OpenCV's
    optimized implementation when available, or falls back to a high-quality
    pure NumPy implementation otherwise.
    
    The function is specifically designed for upsampling nearest neighbor fields
    in the context of pyramid-based PatchMatch algorithms. It handles coordinate
    scaling correctly to ensure that the upsampled coordinates map accurately
    to the new image dimensions.
    
    :param nnf: Nearest neighbor field to upsample with shape (H, W, 2)
    :type nnf: numpy.ndarray
    :param new_height: Target height for the upsampled NNF
    :type new_height: int
    :param new_width: Target width for the upsampled NNF  
    :type new_width: int
    :return: Upsampled NNF as int32 array with shape (new_height, new_width, 2)
    :rtype: numpy.ndarray
    """
    if HAS_OPENCV:
        return _upsample_opencv(nnf, new_height, new_width)
    else:
        return _upsample_numpy(nnf, new_height, new_width) 