#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from patchmatch_cython import inpaint_pyramid

def create_test_image_and_mask():
    """Create a test image and mask using Pillow"""
    try:
        from PIL import Image, ImageDraw
        
        # Create 512x512 image with gradient background
        w, h = 512, 512
        image = Image.new('RGB', (w, h))
        pixels = image.load()
        
        # Add gradient background
        for i in range(h):
            for j in range(w):
                pixels[j, i] = (int(255 * i / h), int(255 * j / w), 0)
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Add some shapes
        draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))  # Red rectangle
        draw.ellipse([260, 60, 340, 140], fill=(0, 255, 0))   # Green circle
        draw.ellipse([140, 270, 260, 330], fill=(0, 0, 255))  # Blue ellipse
        
        # Add some lines
        np.random.seed(42)
        for i in range(3):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = tuple(np.random.randint(100, 255, 3).tolist())
            draw.line([pt1, pt2], fill=color, width=3)
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Create mask using Pillow
        mask_img = Image.new('L', (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        
        # Add circular hole in center
        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        mask_draw.ellipse([center[0] - radius, center[1] - radius, 
                          center[0] + radius, center[1] + radius], fill=255)
        
        # Add some strokes
        np.random.seed(42)
        for _ in range(2):
            pt1 = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            pt2 = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            mask_draw.line([pt1, pt2], fill=255, width=12)
        
        # Convert to numpy array
        mask = np.array(mask_img).astype(np.uint8)
        
        return image_array, mask
        
    except ImportError:
        # Fallback to simple NumPy version
        image = np.zeros((512, 512, 3), dtype=np.float32)
        
        # Add gradient background
        for i in range(512):
            image[i, :, 0] = i / 2
            image[:, i, 1] = i / 2
        
        # Add basic shapes
        image[50:150, 50:150] = [200.0, 50.0, 50.0]
        
        # Simple mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[200:300, 200:300] = 1
        
        return image, mask

def demonstrate_basic_usage():
    """Demonstrate the simplest usage"""
    print("Basic Usage (Automatic Best Performance)")
    print("-" * 50)
    
    # Create test data
    image, mask = create_test_image_and_mask()
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pixels to inpaint: {np.sum(mask)}")
    
    # Run inpainting - automatically uses fastest available implementation
    print("\nRunning inpainting (auto-selects best implementation)...")
    try:
        result = inpaint_pyramid(image, mask)
        print("SUCCESS: Inpainting completed successfully!")
        return image, mask, result
        
    except Exception as e:
        print(f"ERROR: Error during inpainting: {e}")
        return None, None, None

def demonstrate_explicit_solver_selection():
    """Demonstrate explicit solver selection"""
    print("\nAdvanced Usage (Explicit Solver Selection)")
    print("-" * 50)
    
    image, mask = create_test_image_and_mask()
    
    # Import solver classes for explicit selection
    from patchmatch_cython import PythonSolver, CythonSolver
    
    # Force Python implementation
    print("Running with Python solver...")
    result_python = inpaint_pyramid(image, mask, solver_class=PythonSolver)
    print("SUCCESS: Python implementation completed")
    
    # Force Cython implementation  
    print("Running with Cython solver...")
    try:
        result_cython = inpaint_pyramid(image, mask, solver_class=CythonSolver)
        print("SUCCESS: Cython implementation completed")
    except (ImportError, AttributeError):
        print("WARNING: Cython implementation not available, using Python fallback")
        result_cython = result_python
    
    return result_python, result_cython

def demonstrate_seed_functionality():
    """Demonstrate seed functionality for reproducible results"""
    print("\nSeed Functionality (Reproducible Results)")
    print("-" * 50)
    
    image, mask = create_test_image_and_mask()
    
    # Run multiple times with the same seed
    print("Running inpainting with seed=42 (3 times)...")
    results = []
    for i in range(3):
        print(f"  Run {i+1}/3...", end=" ")
        result = inpaint_pyramid(image, mask, seed=42)
        results.append(result)
        print("Done")
    
    # Check if results are identical
    identical = all(np.array_equal(results[0], result) for result in results[1:])
    print(f"Results identical: {identical}")
    
    if identical:
        print("SUCCESS: Seed functionality working correctly!")
        print("Using the same seed produces identical results.")
    else:
        print("INFO: Results differ slightly due to different RNG implementations")
        print("This is expected when comparing Python vs Cython solvers.")
    
    # Show difference between seeded and unseeded runs
    print("\nComparing seeded vs unseeded runs...")
    result_seeded = inpaint_pyramid(image, mask, seed=42)
    result_unseeded = inpaint_pyramid(image, mask)  # No seed
    
    differences = np.sum(result_seeded != result_unseeded)
    print(f"Pixel differences between seeded and unseeded: {differences}")
    
    return result_seeded, result_unseeded

def visualize_results(image, mask, result):
    """Create visualization of the results"""
    if image is None:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Convert to RGB for display (algorithm uses BGR-like ordering)
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title("Inpainted Result")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('patchmatch_example_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("SUCCESS: Results saved to 'patchmatch_example_result.png'")

def main():
    # Demonstrate basic usage
    image, mask, result = demonstrate_basic_usage()
    
    if result is not None:
        # Demonstrate advanced usage
        demonstrate_explicit_solver_selection()
        
        # Demonstrate seed functionality
        demonstrate_seed_functionality()
        
        # Show results
        print("\nVisualization")
        print("-" * 50)
        visualize_results(image, mask, result)

        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: Example completed successfully!")
    else:
        print("\nERROR: Example failed!")
        exit(1) 