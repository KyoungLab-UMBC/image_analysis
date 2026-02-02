
import os
import glob
print("--- LAUNCHING SCRIPT ---")  # <--- ADD THIS AT LINE 1
import numpy as np
import tifffile
print("--- IMPORTING NUMPY ---")   # <--- ADD THIS
from scipy.ndimage import gaussian_laplace
print("--- IMPORTING SCIPY ---")   # <--- ADD THIS
from skimage import measure, morphology
from skimage.restoration import denoise_tv_chambolle 
print("--- IMPORTING SKIMAGE ---") # <--- ADD THIS
# ... rest of your code ...

def dot_2d(struct_img, log_sigma, cutoff=-1):
    """Apply 2D spot filter (LoG)"""
    # Check dimensions safely
    if struct_img.ndim != 2:
        print(f"Warning: dot_2d received {struct_img.ndim}D image")
    
    response = -1 * (log_sigma**2) * gaussian_laplace(struct_img, log_sigma)
    if cutoff < 0:
        return response
    else:
        return response > cutoff

def edge_preserving_smoothing_2d(struct_img, weight=0.1):
    """
    Replaced ITK with Scikit-Image Total Variation Denoising.
    This is stable and won't crash the kernel.
    """
    return denoise_tv_chambolle(struct_img, weight=weight)

def normalize_fast(img, check_rate=100):
    stride = int(np.sqrt(check_rate))
    if stride < 1: stride = 1
    subsample = img[::stride, ::stride]
    min_val = np.min(subsample)
    max_val = np.max(subsample)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.float32)
    norm_img = (img - min_val) / (max_val - min_val)
    return np.clip(norm_img, 0, 1)

def filter_shape(mask, max_area, max_ar):
    labeled_img = measure.label(mask)
    regions = measure.regionprops(labeled_img)
    new_mask = np.zeros_like(mask, dtype=bool)
    
    for region in regions:
        if region.area > max_area: continue
        
        if region.minor_axis_length == 0:
            ar = 9999.0
        else:
            ar = region.major_axis_length / region.minor_axis_length
            
        if ar > max_ar: continue
            
        coords = region.coords
        new_mask[coords[:,0], coords[:,1]] = True
        
    return new_mask

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob.glob(os.path.join(input_folder, "*.tif"))
    print(f"Found {len(files)} images in {input_folder}")

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}...")
        
        try:
            raw_img = tifffile.imread(filepath)
            
            # Debug Print
            print(f"  > Image loaded. Shape: {raw_img.shape}")

            norm_img = normalize_fast(raw_img, check_rate=100)
            
            # Use the new safe smoothing
            smoothed = edge_preserving_smoothing_2d(norm_img, weight=0.1)
            print("  > Smoothing done.")
            
            # --- Dot Detection ---
            mask_a = dot_2d(smoothed, log_sigma=4.5, cutoff=0.2)
            mask_b = dot_2d(smoothed, log_sigma=3.0, cutoff=0.15)
            mask_c = dot_2d(smoothed, log_sigma=4.5, cutoff=0.2)
            
            raw_mask_d = dot_2d(smoothed, log_sigma=1.5, cutoff=0.06)
            mask_d = filter_shape(raw_mask_d, max_area=50, max_ar=2.5)
            
            combined_mask = mask_a | mask_b | mask_c | mask_d
            final_mask = morphology.remove_small_objects(combined_mask, min_size=5)
            
            save_img = (final_mask * 255).astype(np.uint8)
            
            save_name = f"{os.path.splitext(filename)[0]}_mask.tif"
            save_path = os.path.join(output_folder, save_name)
            
            tifffile.imwrite(save_path, save_img)
            print(f"  > Saved to: {save_name}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_dir = r"F:\20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 1 - 140 mM\Cell3\input"
    output_dir = r"F:\20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 1 - 140 mM\Cell3\output"
    
    process_folder(input_dir, output_dir)