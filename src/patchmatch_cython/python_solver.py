import typing

import numpy as np

from .solver import Solver

# Type alias for coordinates that can be tuple, list, or ndarray
Coords = typing.Union[typing.Tuple[int, int], typing.List[int], np.ndarray]


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
    """

    def __init__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            patch_size: int
    ) -> None:

        super().__init__(image, mask, patch_size)

        self._nnf: np.ndarray | None = None
        self._masked_coords: np.ndarray = np.array(np.where(mask == True)).T
        self._is_valid_coords: np.ndarray = np.logical_not(mask)
        self._valid_coords: np.ndarray = np.array(np.where(self._is_valid_coords)).T

    @property
    def nnf(self) -> np.ndarray:
        return self._nnf

    @nnf.setter
    def nnf(self, value: np.ndarray):
        self._nnf = value

    def _bounds_check_coord(self, x: int, y: int) -> bool:
        """
        Check if coordinates are within image bounds.

        Validates that the given (x, y) coordinates fall within the valid
        image dimensions, ensuring they can be used for pixel access.

        :param x: X coordinate (column index)
        :type x: int
        :param y: Y coordinate (row index)
        :type y: int
        :return: True if coordinates are within bounds, False otherwise
        :rtype: bool
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def _bounds_check_patch(self, x: int, y: int) -> bool:
        """
        Check if a patch centered at given coordinates fits within image bounds.

        Validates that a square patch of size ``patch_size`` centered at the
        given coordinates would lie entirely within the image boundaries.

        :param x: X coordinate of patch center
        :type x: int
        :param y: Y coordinate of patch center
        :type y: int
        :return: True if patch fits within image bounds, False otherwise
        :rtype: bool
        """
        return (x - self.half_patch_size >= 0 and
                y - self.half_patch_size >= 0 and
                x + self.half_patch_size < self.width and
                y + self.half_patch_size < self.height)

    def _get_patch(self, image: np.ndarray, coords: Coords) -> np.ndarray:
        """
        Extract a square patch from an image, padding with zeros if out of bounds.

        Extracts a patch of size ``patch_size`` × ``patch_size`` centered at the
        given coordinates. If the patch extends beyond image boundaries, it is
        padded with zeros to maintain the correct patch dimensions.

        :param image: Source image to extract patch from
        :type image: numpy.ndarray
        :param coords: Center coordinates of the patch as (x, y)
        :type coords: Coords
        :return: Extracted patch with shape (patch_size, patch_size, channels)
        :rtype: numpy.ndarray
        """
        x, y = coords
        h = self.half_patch_size

        # Calculate slice bounds
        y1, y2 = max(0, y - h), min(image.shape[0], y + h + 1)
        x1, x2 = max(0, x - h), min(image.shape[1], x + h + 1)

        # Extract and pad if needed
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
        """
        Compute sum of squared differences between two patches.

        Calculates the L2 distance between two patches extracted from different
        images at the specified coordinates. Patches are automatically padded
        with zeros if they extend beyond image boundaries.

        :param patch_image1: First source image
        :type patch_image1: numpy.ndarray
        :param patch_coords1: Center coordinates for first patch as (x, y)
        :type patch_coords1: Coords
        :param patch_image2: Second source image
        :type patch_image2: numpy.ndarray
        :param patch_coords2: Center coordinates for second patch as (x, y)
        :type patch_coords2: Coords
        :return: Sum of squared differences between the patches
        :rtype: float
        """

        patch1: np.ndarray = self._get_patch(patch_image1, patch_coords1)
        patch2: np.ndarray = self._get_patch(patch_image2, patch_coords2)

        return np.sum((patch1 - patch2) ** 2)

    def _randomize_nnf(self) -> None:
        """
        Initialize the nearest neighbor field (NNF) with random valid coordinates.

        Creates a nearest neighbor field where each masked pixel is assigned
        a random valid (unmasked) coordinate from the source image. Non-masked
        pixels are assigned their own coordinates as identity mappings.

        The NNF is stored as a 3D array with shape (height, width, 2) where
        the last dimension contains (x, y) coordinates.

        :return: None (modifies ``self._nnf`` in place)
        :rtype: None
        """
        h, w = self.height, self.width
        self._nnf = np.zeros((h, w, 2), dtype=np.int32)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Default: identity mapping (non-masked pixels)
        self._nnf[..., 0] = x_coords
        self._nnf[..., 1] = y_coords

        # Override with random coords for masked pixels
        masked_pixels = np.where(self.mask)
        if len(masked_pixels[0]) > 0:
            random_indices = np.random.choice(len(self._valid_coords), size=len(masked_pixels[0]))
            random_coords = self._valid_coords[random_indices]
            self._nnf[masked_pixels] = random_coords[:, ::-1]  # Store as (x, y)

    def _paint_with_nnf(self, target: np.ndarray) -> None:
        """
        Paint the target image using the current nearest neighbor field.

        Fills in the masked regions of the target image by copying pixel values
        from the source image according to the nearest neighbor mappings stored
        in the NNF. Each masked pixel is replaced with the pixel value from its
        matched coordinate in the source image.

        :param target: Target image to paint (modified in place)
        :type target: numpy.ndarray
        :return: None (modifies target image in place)
        :rtype: None
        """

        if len(self._masked_coords) == 0:
            return

        # Get all masked coordinates and their NNF matches
        masked_y, masked_x = self._masked_coords[:, 0], self._masked_coords[:, 1]
        matches = self._nnf[masked_y, masked_x]
        match_x, match_y = matches[:, 0], matches[:, 1]

        # Copy pixels from source to target
        target[masked_y, masked_x] = self.image[match_y, match_x]

    def __call__(
            self,
            iterations: int = 5,
            randomize_nnf: bool = True,
            nnf: typing.Optional[np.ndarray] = None,
            seed: typing.Optional[int] = None
    ) -> np.ndarray:
        """
        Perform image inpainting using the PatchMatch algorithm.

        Executes the main PatchMatch algorithm to fill in masked regions of the
        source image. The algorithm alternates between forward and backward passes,
        using propagation and random search to iteratively improve the nearest
        neighbor field and reconstruct the inpainted image.

        The algorithm performs the following steps:
        1. Initialize the nearest neighbor field (NNF)
        2. For each iteration:
           - Propagate good matches from neighboring pixels
           - Perform random search to escape local minima
           - Reconstruct the inpainted image using current NNF
        3. Return the final inpainted result

        :param iterations: Number of PatchMatch iterations to perform
        :type iterations: int
        :param randomize_nnf: Whether to randomly initialize the NNF. If False,
                          must provide ``nnf`` parameter
        :type randomize_nnf: bool
        :param nnf: Pre-computed nearest neighbor field to use instead of
                   random initialization. Shape must be (height, width, 2)
        :type nnf: numpy.ndarray or None
        :param seed: Random seed for this specific run (overrides instance seed)
        :type seed: int or None
        :return: Inpainted image with masked regions filled
        :rtype: numpy.ndarray
        """

        if seed is not None:
            np.random.seed(seed)

        inpainted_image = self.image.copy()

        # Initialize NNF
        if randomize_nnf:
            self._randomize_nnf()
        else:
            self._nnf = nnf

        self._paint_with_nnf(inpainted_image)

        # Precompute constants
        max_search_radius = max(self.height, self.width)
        neighbor_offsets = [np.array([(-1, 0), (0, -1)]), np.array([(1, 0), (0, 1)])]

        for it in range(iterations):
            # Determine scan order and neighbor directions
            is_forward = it % 2 == 0
            coords_order = self._masked_coords if is_forward else reversed(self._masked_coords)
            neighbors = neighbor_offsets[0] if is_forward else neighbor_offsets[1]

            for y, x in coords_order:
                best_match = self._nnf[y, x]
                best_distance = self._patch_distance(inpainted_image, [x, y], self.image, best_match)

                # Propagation phase
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx

                    if not self._bounds_check_coord(nx, ny):
                        continue

                    # Calculate propagated candidate
                    candidate = self._nnf[ny, nx] - np.array([dx, dy])
                    cx, cy = candidate[0].item(), candidate[1].item()

                    if (self._bounds_check_coord(cx, cy) and
                            self._is_valid_coords[cy, cx]):

                        distance = self._patch_distance(inpainted_image, [x, y], self.image, candidate)
                        if distance < best_distance:
                            best_match = candidate
                            best_distance = distance

                # Random search phase
                search_radius = max_search_radius
                bx, by = best_match[0].item(), best_match[1].item()

                while search_radius > 1:
                    # Generate random candidate
                    rx = bx + np.random.randint(-search_radius, search_radius + 1)
                    ry = by + np.random.randint(-search_radius, search_radius + 1)

                    if (self._bounds_check_coord(rx, ry) and
                            self._is_valid_coords[ry, rx]):

                        candidate = np.array([rx, ry])
                        distance = self._patch_distance(inpainted_image, [x, y], self.image, candidate)

                        if distance < best_distance:
                            best_match = candidate
                            best_distance = distance
                            bx, by = rx, ry  # Update center for next iteration

                    search_radius //= 2

                self._nnf[y, x] = best_match

            self._paint_with_nnf(inpainted_image)

        return inpainted_image
