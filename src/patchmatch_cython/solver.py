import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class Solver(ABC):
    """
    Abstract base class for PatchMatch solvers.
    
    This class defines the common interface that all PatchMatch solver implementations
    must follow. It handles basic initialization and provides the abstract method
    signature for the main algorithm execution.
    
    All concrete solver implementations inherit from this class and must implement
    the :meth:`run` method to provide their specific algorithm implementation.
    
    :param image: Input image array with shape (H, W, C)
    :type image: numpy.ndarray
    :param mask: Binary mask indicating areas to inpaint with shape (H, W)
    :type mask: numpy.ndarray
    :param patch_size: Size of square patches to use for matching
    :type patch_size: int
    """
    
    def __init__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            patch_size: int
    ) -> None:
        """
        Initialize the solver with source image, mask, and patch size.
        
        Sets up the common attributes needed by all solver implementations,
        including image dimensions, coordinates of masked pixels, and the
        nearest neighbor field placeholder.
        
        :param image: Input image array with shape (H, W, C)
        :type image: numpy.ndarray
        :param mask: Binary mask indicating areas to inpaint with shape (H, W)
        :type mask: numpy.ndarray
        :param patch_size: Size of square patches to use for matching
        :type patch_size: int
        """

        self.image: np.ndarray = image
        self.mask: np.ndarray = mask
        self.patch_size: int = patch_size
        self.half_patch_size: int = patch_size // 2
        self.height: int
        self.width: int
        self.height, self.width = image.shape[:2]
    
    @abstractmethod
    def __call__(
            self,
            iterations: int = 5,
            randomize_nnf: bool = True,
            nnf: Optional[np.ndarray] = None,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Run the PatchMatch algorithm.
        
        This abstract method must be implemented by all concrete solver classes.
        It performs the main PatchMatch algorithm execution and returns the
        inpainted image result.
        
        :param iterations: Number of PatchMatch iterations to perform
        :type iterations: int
        :param randomize_nnf: Whether to use random initialization for the nearest neighbor field
        :type randomize_nnf: bool
        :param nnf: Pre-computed nearest neighbor field to use for initialization
                   (useful for pyramid approach). Shape must be (height, width, 2)
        :type nnf: numpy.ndarray or None
        :param seed: Random seed for this specific run, overrides instance seed if provided
        :type seed: int or None
        :return: Inpainted image as numpy array with same shape as input
        :rtype: numpy.ndarray
        """
        pass

    @property
    @abstractmethod
    def nnf(self) -> np.ndarray:
        pass

    @nnf.setter
    @abstractmethod
    def nnf(self, value: np.ndarray):
        pass