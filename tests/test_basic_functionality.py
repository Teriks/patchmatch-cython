"""
Basic tests for patchmatch_cython
"""

import pytest
import numpy as np

def test_import_core_algorithm():
    """Test that the core algorithm can be imported"""
    from patchmatch_cython.python_solver import PythonSolver
    from patchmatch_cython.inpaint import inpaint_pyramid, inpaint_single
    assert callable(inpaint_pyramid)
    assert callable(inpaint_single)
    assert PythonSolver is not None

def test_import_cython_acceleration():
    """Test that Cython acceleration can be imported"""
    try:
        from patchmatch_cython.cython_solver import CythonSolver
        assert CythonSolver is not None
    except ImportError:
        pytest.skip("Cython acceleration not available")

def test_import_cython_module():
    """Test that the Cython module can be imported"""
    try:
        import patchmatch_cython.cython_solver_impl
        assert hasattr(patchmatch_cython.cython_solver_impl, 'CythonSolverImpl')
    except ImportError:
        pytest.skip("cython_solver_impl extension not built")

def test_basic_functionality():
    """Test basic functionality with small image"""
    from patchmatch_cython.inpaint import inpaint_single
    
    # Create small test image
    image = np.random.rand(50, 50, 3).astype(np.float32) * 255
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    
    # Test single resolution (preserves input size)
    result = inpaint_single(image, mask)
    assert result.shape == image.shape
    assert result.dtype == np.uint8

def test_cython_functionality():
    """Test Cython functionality with small image"""
    try:
        from patchmatch_cython.inpaint import inpaint_single
        from patchmatch_cython.cython_solver import CythonSolver
        
        # Create small test image
        image = np.random.rand(50, 50, 3).astype(np.float32) * 255
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 1
        
        # Test single resolution with explicit Cython solver
        result = inpaint_single(image, mask, solver_class=CythonSolver)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    except ImportError:
        pytest.skip("Cython acceleration not available")

def test_package_version():
    """Test that package has a version"""
    try:
        import patchmatch_cython
        assert hasattr(patchmatch_cython, '__version__')
        assert isinstance(patchmatch_cython.__version__, str)
    except ImportError:
        pytest.skip("patchmatch_cython not available") 