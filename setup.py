from setuptools import setup, Extension, find_packages
import numpy as np
import os
import platform

# Always require Cython and .pyx source
pyx_file = os.path.join("src", "patchmatch_cython", "cython_solver_impl.pyx")
if not os.path.exists(pyx_file):
    raise RuntimeError(f"cython_solver_impl.pyx source file not found at {pyx_file}!")

try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython is required to build this package. Install with: pip install Cython")

def macos_target_archs():
    """Return the macOS architectures this compilation is targeting.

    cibuildwheel cross-compiles by setting ARCHFLAGS (for example
    ``-arch x86_64``) on an Apple Silicon runner. platform.machine() still
    reports the host CPU in that case, so it cannot be used to choose flags.
    """
    archs = []
    parts = os.environ.get("ARCHFLAGS", "").split()
    for index, part in enumerate(parts):
        if part == "-arch" and index + 1 < len(parts):
            archs.append(parts[index + 1])
    if archs:
        return archs

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return ["arm64"]
    if machine in ("x86_64", "amd64"):
        return ["x86_64"]
    return [machine]


def get_compiler_args():
    """Get platform-appropriate compiler arguments"""
    if os.name == 'nt':  # Windows
        return ["/O2"]
    elif platform.system() == 'Darwin':  # macOS
        archs = macos_target_archs()
        # -mcpu=apple-m1 is valid only for arm64. cibuildwheel's x86_64 builds
        # run on Apple Silicon runners, and passing that flag makes clang fail
        # with "unsupported option '-mcpu='".
        if archs == ["arm64"]:
            return ["-O3", "-mcpu=apple-m1"]
        # -march=native is only safe when actually compiling on Intel.
        if archs == ["x86_64"] and platform.machine().lower() in ("x86_64", "amd64"):
            return ["-O3", "-march=native"]
        return ["-O3"]
    else:  # Linux and other Unix-like
        return ["-O3", "-march=native"]

# Build from Cython source only
extensions = [
    Extension(
        "patchmatch_cython.cython_solver_impl",
        [pyx_file],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=get_compiler_args(),
    )
]

ext_modules = cythonize(
    extensions, 
    compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True,
    }
)

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="patchmatch-cython",
    version="0.1.5",
    author="Teriks",
    author_email="Teriks999@gmail.com",
    description="High-performance PatchMatch implementation for image inpainting using Cython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Teriks/patchmatch-cython",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "visualization": [
            "matplotlib>=3.5.0",
            "Pillow>=8.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "build>=0.8.0",
            "cibuildwheel>=3.0.1",
            "setuptools>=61.0",
            "wheel",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)

# Build with: python setup.py build_ext --inplace 
# Or: pip install -e .
# Or: python -m build 