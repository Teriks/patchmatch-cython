import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List

# Try to import both solver implementations from top level
try:
    from patchmatch_cython import PythonSolver
    PYTHON_AVAILABLE = True
except ImportError:
    print("Python solver not available")
    PYTHON_AVAILABLE = False

try:
    from patchmatch_cython import CythonSolver
    CYTHON_AVAILABLE = True
except ImportError:
    print("Cython solver not available")
    CYTHON_AVAILABLE = False

def create_test_image(size=(512, 512), seed: int = 42) -> np.ndarray:
    """Create a synthetic test image with various features"""
    np.random.seed(seed)
    h, w = size
    
    try:
        from PIL import Image, ImageDraw
        
        # Create image with gradient background
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
        for i in range(5):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = tuple(np.random.randint(100, 255, 3).tolist())
            draw.line([pt1, pt2], fill=color, width=2)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Add texture noise
        noise = np.random.randint(0, 30, (h, w, 3))
        image_array = np.clip(image_array.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        return image_array
        
    except ImportError:
        # Fallback to basic NumPy if Pillow not available
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(h):
            image[i, :, 0] = int(255 * i / h)
            image[:, i, 1] = int(255 * i / w)
        
        # Add basic shapes
        image[50:150, 50:150] = [255, 0, 0]  # Red rectangle
        
        # Add circle (approximate)
        y, x = np.ogrid[:h, :w]
        circle_mask = (x - 300)**2 + (y - 100)**2 <= 40**2
        image[circle_mask] = [0, 255, 0]
        
        # Add texture noise
        noise = np.random.randint(0, 30, (h, w, 3))
        image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        return image

def create_test_mask(image_shape, seed: int = 42) -> np.ndarray:
    """Create a test mask with multiple holes"""
    np.random.seed(seed)
    h, w = image_shape[:2]
    
    try:
        from PIL import Image, ImageDraw
        
        # Create mask image
        mask_img = Image.new('L', (w, h), 0)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(mask_img)
        
        # Add circular hole in center
        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        draw.ellipse([center[0] - radius, center[1] - radius, 
                     center[0] + radius, center[1] + radius], fill=255)
        
        # Add some random strokes
        for _ in range(3):
            # Random thick line
            pt1 = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            pt2 = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            draw.line([pt1, pt2], fill=255, width=10)
        
        # Convert to numpy boolean array
        mask = np.array(mask_img) > 0
        
        return mask
        
    except ImportError:
        # Fallback to NumPy if Pillow not available
        mask = np.zeros((h, w), dtype=bool)
        
        # Add circular hole in center
        center = (w // 2, h // 2)
        radius = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask |= (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Add some random strokes (simple version)
        for _ in range(3):
            y_start = np.random.randint(h//4, 3*h//4)
            x_start = np.random.randint(w//4, 3*w//4)
            y_end = np.random.randint(h//4, 3*h//4)
            x_end = np.random.randint(w//4, 3*w//4)
            
            # Draw thick line
            steps = max(abs(x_end - x_start), abs(y_end - y_start))
            if steps > 0:
                for step in range(steps):
                    t = step / steps
                    y = int(y_start + t * (y_end - y_start))
                    x = int(x_start + t * (x_end - x_start))
                    
                    # Make line thick (thickness=10)
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                mask[ny, nx] = True
        
        return mask

def benchmark_solver(solver_class, image: np.ndarray, mask: np.ndarray, 
                    patch_size: int = 5, iterations: int = 5, 
                    seed: int = 42, runs: int = 3) -> Dict[str, Any]:
    """Benchmark a solver implementation using the proper inpainting functions"""
    
    times = []
    result_image = None
    
    # Import the inpainting functions
    from patchmatch_cython import inpaint_pyramid
    
    for run in range(runs):
        print(f"  Run {run + 1}/{runs}...", end=" ")
        
        # Time the inpainting using the high-level API
        start_time = time.time()
        
        # Use pyramid inpainting for better quality (default approach)
        result = inpaint_pyramid(
            image=image.copy(), 
            mask=mask.copy(), 
            patch_size=patch_size, 
            solver_class=solver_class,
            seed=seed
        )
        
        end_time = time.time()
        
        runtime = end_time - start_time
        times.append(runtime)
        print(f"{runtime:.3f}s")
        
        # Store result from first run for visualization
        if run == 0:
            result_image = result.copy()
    
    return {
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_image': result_image
    }

def create_comprehensive_visualization(all_results: Dict[str, Dict[str, Any]], 
                                     image_size: Tuple[int, int]) -> None:
    """Create comprehensive visualization of all results"""
    
    # Set up the figure
    patch_sizes = list(all_results.keys())
    solver_names = list(all_results[patch_sizes[0]].keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
    
    # Display original image and mask (first row, first 2 columns)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_mask = fig.add_subplot(gs[0, 1])
    
    # Get original image and mask from first result
    first_patch = patch_sizes[0]
    first_solver = solver_names[0]
    
    # We'll need to recreate the image for display
    image = create_test_image(image_size)
    mask = create_test_mask(image.shape)
    
    ax_orig.imshow(image)
    ax_orig.set_title(f'Original Image ({image_size[0]}x{image_size[1]})')
    ax_orig.axis('off')
    
    ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title('Mask')
    ax_mask.axis('off')
    
    # Display results for each patch size
    col_offset = 2
    for i, patch_size in enumerate(patch_sizes):
        for j, solver_name in enumerate(solver_names):
            if solver_name in all_results[patch_size]:
                ax = fig.add_subplot(gs[0, col_offset + i * 2 + j])
                result_img = all_results[patch_size][solver_name]['result_image']
                ax.imshow(result_img)
                ax.set_title(f'{solver_name}\nPatch Size {patch_size}')
                ax.axis('off')
    
    # Performance comparison chart (second row)
    ax_perf = fig.add_subplot(gs[1, :3])
    
    # Collect performance data
    x_labels = []
    python_times = []
    cython_times = []
    
    for patch_size in patch_sizes:
        x_labels.append(f'Patch {patch_size}')
        if 'Python' in all_results[patch_size]:
            python_times.append(all_results[patch_size]['Python']['mean_time'])
        else:
            python_times.append(0)
            
        if 'Cython' in all_results[patch_size]:
            cython_times.append(all_results[patch_size]['Cython']['mean_time'])
        else:
            cython_times.append(0)
    
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    bars1 = ax_perf.bar(x_pos - width/2, python_times, width, label='Python', color='lightcoral')
    bars2 = ax_perf.bar(x_pos + width/2, cython_times, width, label='Cython', color='skyblue')
    
    ax_perf.set_xlabel('Configuration')
    ax_perf.set_ylabel('Time (seconds)')
    ax_perf.set_title('Performance Comparison')
    ax_perf.set_xticks(x_pos)
    ax_perf.set_xticklabels(x_labels)
    ax_perf.legend()
    ax_perf.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')
    
    # Speedup analysis (second row, right side)
    ax_speedup = fig.add_subplot(gs[1, 3:])
    
    speedups = []
    for i, patch_size in enumerate(patch_sizes):
        if ('Python' in all_results[patch_size] and 
            'Cython' in all_results[patch_size] and 
            all_results[patch_size]['Cython']['mean_time'] > 0):
            speedup = all_results[patch_size]['Python']['mean_time'] / all_results[patch_size]['Cython']['mean_time']
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    bars = ax_speedup.bar(x_labels, speedups, color='lightgreen')
    ax_speedup.set_xlabel('Configuration')
    ax_speedup.set_ylabel('Speedup Factor')
    ax_speedup.set_title('Cython Speedup')
    ax_speedup.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_speedup.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}x', ha='center', va='bottom')
    
    # Summary statistics (third row)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Create summary text
    summary_text = "Performance Summary\n" + "="*50 + "\n\n"
    
    for patch_size in patch_sizes:
        summary_text += f"Patch Size {patch_size}:\n"
        
        for solver_name in solver_names:
            if solver_name in all_results[patch_size]:
                stats = all_results[patch_size][solver_name]
                summary_text += f"  {solver_name}: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s\n"
        
        # Add speedup
        if ('Python' in all_results[patch_size] and 
            'Cython' in all_results[patch_size] and 
            all_results[patch_size]['Cython']['mean_time'] > 0):
            speedup = all_results[patch_size]['Python']['mean_time'] / all_results[patch_size]['Cython']['mean_time']
            summary_text += f"  Speedup: {speedup:.1f}x\n"
        
        summary_text += "\n"
    
    ax_stats.text(0.05, 0.95, summary_text, transform=ax_stats.transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'PatchMatch Performance Analysis - {image_size[0]}x{image_size[1]} Image', 
                 fontsize=16, y=0.98)

    plt.show()

def run_comprehensive_comparison(image_size: Tuple[int, int] = (512, 512), 
                                patch_sizes: List[int] = [5, 7],
                                iterations: int = 5,
                                runs: int = 3,
                                seed: int = 42) -> None:
    """Run comprehensive comparison across multiple patch sizes"""
    
    print(f"Running comprehensive comparison - {image_size[0]}x{image_size[1]} image")
    print("=" * 60)
    
    # Check available solvers
    available_solvers = []
    if PYTHON_AVAILABLE:
        available_solvers.append(('Python', PythonSolver))
    if CYTHON_AVAILABLE:
        available_solvers.append(('Cython', CythonSolver))
    
    if not available_solvers:
        print("ERROR: No solvers available!")
        return
    
    print(f"Available solvers: {[name for name, _ in available_solvers]}")
    print(f"Testing patch sizes: {patch_sizes}")
    print(f"Iterations: {iterations}, Runs per test: {runs}")
    print()
    
    # Create test data
    image = create_test_image(image_size, seed)
    mask = create_test_mask(image.shape, seed)
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pixels to inpaint: {np.sum(mask)}")
    print()
    
    # Run tests for each configuration
    all_results = {}
    
    for patch_size in patch_sizes:
        print(f"Testing patch size {patch_size}")
        print("-" * 40)
        
        all_results[patch_size] = {}
        
        for solver_name, solver_class in available_solvers:
            print(f"Running {solver_name} solver...")
            
            try:
                results = benchmark_solver(
                    solver_class, image, mask, 
                    patch_size=patch_size, 
                    iterations=iterations, 
                    seed=seed, 
                    runs=runs
                )
                all_results[patch_size][solver_name] = results
                print(f"  Average time: {results['mean_time']:.2f}s ± {results['std_time']:.2f}s")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        print()
    
    # Create visualization
    create_comprehensive_visualization(all_results, image_size)

def main():
    """Main function"""
    print("PatchMatch Performance Comparison")
    print("=" * 50)
    
    if not PYTHON_AVAILABLE and not CYTHON_AVAILABLE:
        print("ERROR: No implementations available!")
        return
    
    # Run comprehensive comparison
    run_comprehensive_comparison(
        image_size=(512, 512),
        patch_sizes=[5, 7],
        iterations=5,
        runs=3,
        seed=42
    )

if __name__ == "__main__":
    main() 