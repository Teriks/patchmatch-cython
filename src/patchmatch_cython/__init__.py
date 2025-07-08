import typing

__version__: str = "0.1.4"

# Core algorithm imports (always available)
from .python_solver import PythonSolver
from .inpaint import inpaint_pyramid, inpaint_single

# Try to import Cython-accelerated version
try:
    from .cython_solver import CythonSolver
    __all__: typing.List[str] = [
        "PythonSolver", "inpaint_pyramid", "inpaint_single",
        "CythonSolver"
    ]
except ImportError:
    __all__ = [
        "PythonSolver", "inpaint_pyramid", "inpaint_single"
    ] 