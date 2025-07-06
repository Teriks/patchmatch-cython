#!/usr/bin/env python3
"""
Local wheel building script

This script builds wheels for all supported Python versions using cibuildwheel.
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

def run_command(cmd, env=None):
    """Run command and return success status"""
    print(f"RUNNING: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        return False

def clean_build_artifacts():
    """Clean previous build artifacts"""
    artifacts_to_clean = [
        "wheelhouse",
        "build",
        "dist",
        "*.egg-info",
        "src/**/*.cpp",
        "src/**/*.c",
        "src/**/*.so",
        "src/**/*.pyd",
    ]
    
    print("Cleaning previous build artifacts...")
    for pattern in artifacts_to_clean:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            for path in glob.glob(pattern, recursive=True):
                if os.path.isdir(path):
                    print(f"  Removing directory: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    print(f"  Removing file: {path}")
                    os.remove(path)
        else:
            path = Path(pattern)
            if path.exists():
                if path.is_dir():
                    print(f"  Removing directory: {path}")
                    shutil.rmtree(path)
                else:
                    print(f"  Removing file: {path}")
                    path.unlink()

def build_wheels(python_versions=None, platforms=None, clean=True):
    """Build wheels using cibuildwheel"""
    
    if clean:
        clean_build_artifacts()
    
    # Create custom environment
    env = os.environ.copy()
    
    # Set build configuration
    if python_versions:
        # Build specific Python versions
        if isinstance(python_versions, list):
            build_pattern = " ".join([f"cp{v.replace('.', '')}-*" for v in python_versions])
        else:
            build_pattern = f"cp{python_versions.replace('.', '')}-*"
        env["CIBW_BUILD"] = build_pattern
        print(f"Building for Python versions: {python_versions}")
    else:
        # Build all supported versions (default)
        env["CIBW_BUILD"] = "cp310-* cp311-* cp312-* cp313-*"
        print("Building for all supported Python versions (3.10-3.13)")
    

    
    # Set platform-specific configurations
    if platforms:
        if isinstance(platforms, list):
            platform_arg = f"--platform {' '.join(platforms)}"
        else:
            platform_arg = f"--platform {platforms}"
    else:
        platform_arg = ""  # Build for current platform
    
    # Configure test command
    env["CIBW_TEST_COMMAND"] = "python -c \"import patchmatch_cython; print('Import successful')\""
    
    print(f"\n{'='*70}")
    print("BUILDING WHEELS")
    print(f"{'='*70}")
    
    # Run cibuildwheel
    cmd = f"cibuildwheel --output-dir wheelhouse {platform_arg}".strip()
    success = run_command(cmd, env)
    
    if not success:
        # Try as Python module
        cmd = f"{sys.executable} -m cibuildwheel --output-dir wheelhouse {platform_arg}".strip()
        success = run_command(cmd, env)
    
    if success:
        print(f"\n{'='*70}")
        print("BUILD SUCCESSFUL")
        print(f"{'='*70}")
        
        # List built wheels
        wheelhouse = Path("wheelhouse")
        if wheelhouse.exists():
            wheels = list(wheelhouse.glob("*.whl"))
            print(f"\nBuilt {len(wheels)} wheels:")
            for wheel in wheels:
                size_mb = wheel.stat().st_size / (1024 * 1024)
                print(f"  WHEEL: {wheel.name} ({size_mb:.1f} MB)")
        
        return True
    else:
        print(f"\n{'='*70}")
        print("BUILD FAILED")
        print(f"{'='*70}")
        return False

def validate_wheel_installation():
    """Test installing and importing built wheels"""
    wheelhouse = Path("wheelhouse")
    wheels = list(wheelhouse.glob("*.whl"))
    
    if not wheels:
        print("ERROR: No wheels found to test")
        return False
    
    print(f"\n{'='*70}")
    print("TESTING WHEEL INSTALLATION")
    print(f"{'='*70}")
    
    # Test the first wheel
    wheel = wheels[0]
    print(f"Testing wheel: {wheel.name}")
    
    # Install wheel in a temporary way
    test_cmd = f"pip install {wheel} --force-reinstall --no-deps"
    if run_command(test_cmd):
        # Test import
        test_import = "python -c \"import patchmatch_cython; print('SUCCESS: Import successful')\""
        if run_command(test_import):
            print("SUCCESS: Wheel installation and import test passed")
            return True
    
    print("ERROR: Wheel installation test failed")
    return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build wheels for patchmatch-cython")
    parser.add_argument("--python", nargs="*", 
                       help="Python versions to build for (e.g., 3.11 3.12). Default: all supported versions")
    parser.add_argument("--platform", choices=["auto", "linux", "macos", "windows"], 
                       help="Platform to build for. Default: current platform")
    parser.add_argument("--no-clean", action="store_true", 
                       help="Don't clean build artifacts before building")
    parser.add_argument("--test", action="store_true", 
                       help="Test wheel installation after building")
    
    args = parser.parse_args()
    
    print("PatchMatch Cython - Local Wheel Builder")
    print("=" * 70)
    
    # Check if cibuildwheel is installed
    print("Checking for cibuildwheel...")
    cibw_found = False
    
    # Try importing cibuildwheel as a Python module first (most reliable)
    try:
        import cibuildwheel
        print(f"SUCCESS: Found cibuildwheel module version: {cibuildwheel.__version__}")
        cibw_found = True
    except ImportError:
        print("cibuildwheel module not importable, trying command line...")
        
        # Try as a direct command with --help
        try:
            result = subprocess.run(["cibuildwheel", "--help"], check=True, capture_output=True, text=True)
            print("SUCCESS: Found cibuildwheel (direct command)")
            cibw_found = True
        except subprocess.CalledProcessError as e:
            print(f"Direct command failed with exit code {e.returncode}")
        except FileNotFoundError:
            print("Direct command not found")
        
        # Try as a Python module with --help
        if not cibw_found:
            try:
                result = subprocess.run([sys.executable, "-m", "cibuildwheel", "--help"], check=True, capture_output=True, text=True)
                print("SUCCESS: Found cibuildwheel (Python module)")
                cibw_found = True
            except subprocess.CalledProcessError as e:
                print(f"Python module failed with exit code {e.returncode}")
            except FileNotFoundError:
                print("Python module not found")
    
    if not cibw_found:
        print("ERROR: cibuildwheel not found. Install with: pip install cibuildwheel")
        return 1
    
    # Check if we're in the right directory
    if not Path("src/patchmatch_cython/cython_solver_impl.pyx").exists():
        print("ERROR: cython_solver_impl.pyx not found in src/patchmatch_cython/. Run from the project root.")
        print("Current directory:", os.getcwd())
        print("Looking for:", Path("src/patchmatch_cython/cython_solver_impl.pyx").absolute())
        return 1
    
    print("SUCCESS: Environment checks passed")
    
    # Configure build parameters
    python_versions = args.python
    platforms = args.platform
    
    # Build wheels
    success = build_wheels(
        python_versions=python_versions, 
        platforms=platforms, 
        clean=not args.no_clean
    )
    
    if not success:
        return 1
    
    # Test wheel installation if requested
    if args.test:
        validate_wheel_installation()
    
    print(f"\n{'='*70}")
    print("WHEEL BUILDING COMPLETE")
    print(f"{'='*70}")
    print("INFO: Check the 'wheelhouse' directory for built wheels")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 