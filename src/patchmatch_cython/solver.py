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
            patch_size: int,
            edge_weighted: bool = False,
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
        :param edge_weighted: If True, match with boosted gradient channels; if False,
                              use color-only features
        :type edge_weighted: bool
        """

        self.image: np.ndarray = image
        self.mask: np.ndarray = mask
        self.patch_size: int = patch_size
        self.half_patch_size: int = patch_size // 2
        self.edge_weighted: bool = bool(edge_weighted)
        self.height: int
        self.width: int
        self.height, self.width = image.shape[:2]
    
    @abstractmethod
    def __call__(
            self,
            iterations: int = 5,
            randomize_nnf: bool = True,
            nnf: Optional[np.ndarray] = None,
            seed: Optional[int] = None,
            em_loops: int = 1,
            reverse_search: bool = False,
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
        :param em_loops: Outer match→vote→rematch loops (1 preserves prior behavior)
        :type em_loops: int
        :param reverse_search: Enable source→query completeness / reverse search
        :type reverse_search: bool
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