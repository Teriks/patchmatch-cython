import numpy as np
from typing import Union

# Try to import OpenCV for optimized operations
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def downsample(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Downsample an image to the specified dimensions using mean pooling.
    
    This function reduces the image size by dividing it into blocks and computing
    the mean value of each block. This provides a simple but effective downsampling
    approach that preserves important image features while reducing resolution.
    
    The function handles both 2D (grayscale) and 3D (color) images automatically.
    
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
    image = image[:new_height * (h // new_height), :new_width * (w // new_width)]
    image = image.reshape(new_height, h // new_height, new_width, w // new_width, -1)
    return image.mean(axis=(1, 3)).astype(np.float32)


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
    nnf = cv2.resize(nnf, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return np.round(nnf).astype(np.int32)


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
    
    return np.round(result).astype(np.int32)


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