import os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
import background_correction

# ================= CONFIGURATION =================
# 1. Input: Select your path here
ROOT_PATH = r"F:\20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM" 

# 2. Toggle Queen37C Mode
QUEEN37C = True  
# =================================================

def task_generator(root_path, is_queen):
    """
    Yields tasks one by one instead of creating a giant list in memory.
    """
    # 1. Scan for immediate subfolders with prefix "Cell"
    try:
        root_subs = [os.path.join(root_path, d) for d in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, d)) and d.startswith("Cell")]
    except FileNotFoundError:
        return

    # 2. Recursively scan inside those Cell folders
    for cell_folder in root_subs:
        for current_root, _, files in os.walk(cell_folder):
            
            # Filter for TIF files starting with 'Cell' + 4 digits
            tifs = [f for f in files if f.startswith("Cell") and f.endswith(".tif") and f[4:8].isdigit()]
            
            if not tifs:
                continue
            
            tifs.sort()
            first_img = tifs[0]
            try:
                n = int(first_img[4:8])
            except ValueError:
                continue

            path_n1 = os.path.join(current_root, f"Cell{n+1:04d}.tif")
            path_n2 = os.path.join(current_root, f"Cell{n+2:04d}.tif")

            if os.path.exists(path_n1) and os.path.exists(path_n2):
                # YIELD the task data instead of appending to a list
                yield (path_n1, path_n2, current_root, is_queen)

def process_triplet(args):
    """
    Worker function executed by threads.
    Args: (path_n1, path_n2, output_folder, is_queen_mode)
    """
    p1, p2, out_dir, is_queen = args

    try:
        # Read images
        img1 = tifffile.imread(p1)
        img2 = tifffile.imread(p2)

        # ---- Background Correction ----
        # Settings: radius=1000, create=False (subtract), paraboloid=True
        img1_corr = background_correction.estimate_background_rolling_ball(
            img1, radius=1000, create_background=False, use_paraboloid=True
        )
        img2_corr = background_correction.estimate_background_rolling_ball(
            img2, radius=1000, create_background=False, use_paraboloid=True
        )

        # ---- Gaussian Blur (Radius=1) ----
        # Cast to float64 for accurate division later
        blur1 = gaussian_filter(img1_corr.astype(np.float64), sigma=1)
        blur2 = gaussian_filter(img2_corr.astype(np.float64), sigma=1)

        # Prevent division by zero
        blur1[blur1 == 0] = 1e-9
        blur2[blur2 == 0] = 1e-9

        # ---- Ratio Calculation ----
        # img1 is Cell000n+1, img2 is Cell000n+2
        if is_queen:
            # Queen37C=True: Result = (n+2 / n+1) * 10000
            ratio_map = (blur2 / blur1) * 10000
            out_name = "Queen37C_ratiox10000.tif"
        else:
            # Queen37C=False: Result = (n+1 / n+2) * 1000
            ratio_map = (blur1 / blur2) * 1000
            out_name = "Hylight_ratiox1000.tif"

        # Convert to 16-bit integer
        ratio_map = np.clip(ratio_map, 0, 65535).astype(np.uint16)

        # Save Result
        save_path = os.path.join(out_dir, out_name)
        tifffile.imwrite(save_path, ratio_map)
        print(f"[Done] Saved: {save_path}")

    except Exception as e:
        print(f"[Error] Failed in {out_dir}: {e}")

def main():
    background_correction.init_imagej()

    print(f"Scanning root: {ROOT_PATH} ...")

    # Use a generator to fetch tasks on demand
    tasks = task_generator(ROOT_PATH, QUEEN37C)
    
    # Process with limited workers to save RAM
    with ThreadPoolExecutor(max_workers=4) as executor:
        # executor.map consumes the generator automatically
        executor.map(process_triplet, tasks)

    print("All tasks completed.")

if __name__ == "__main__":
    main()