"""Smoke tests for package importability and basic wiring."""

import pytest


def test_import_package():
    import patchmatch_cython

    assert hasattr(patchmatch_cython, "__version__")
    assert isinstance(patchmatch_cython.__version__, str)


def test_import_public_api():
    from patchmatch_cython import PythonSolver, inpaint_pyramid, inpaint_single

    assert callable(inpaint_pyramid)
    assert callable(inpaint_single)
    assert PythonSolver is not None


def test_import_cython_acceleration():
    try:
        from patchmatch_cython import CythonSolver
        from patchmatch_cython import cython_solver_impl
    except ImportError:
        pytest.skip("Cython acceleration not available")

    assert CythonSolver is not None
    assert hasattr(cython_solver_impl, "CythonSolverImpl")
