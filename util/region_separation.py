import numpy as np
import tifffile
from pathlib import Path
from read_roi import read_roi_zip
from scipy.ndimage import gaussian_filter

from util.roi_to_mask import roi_to_mask
import util.config as cfg

def separate_regions_and_bg(img_r, img_g, roi_mask_full):
    """Calculates background and separates the ROI into 4 median-based quadrants."""
    # 1. Get bounding box of the ROI to save processing time
    coords = np.column_stack(np.where(roi_mask_full))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
    else:
        y0, x0, y1, x1 = 0, 0, img_r.shape[0], img_r.shape[1]

    # 2. Run rolling ball only on the bounding box and map back to full size
    r_bg = np.zeros_like(img_r, dtype=float)
    g_bg = np.zeros_like(img_g, dtype=float)
    
    r_bg[y0:y1, x0:x1] = gaussian_filter(img_r[y0:y1, x0:x1].astype(float), sigma=5)
    g_bg[y0:y1, x0:x1] = gaussian_filter(img_g[y0:y1, x0:x1].astype(float), sigma=5)

    # 3. Calculate medians ONLY within the valid ROI mask
    r_bg_med = np.median(r_bg[roi_mask_full])
    g_bg_med = np.median(g_bg[roi_mask_full])

    # 4. Separate into 4 masks (restricted strictly to the ROI)
    mask_hh = (r_bg >= r_bg_med) & (g_bg >= g_bg_med) & roi_mask_full
    mask_lh = (r_bg < r_bg_med)  & (g_bg >= g_bg_med) & roi_mask_full
    mask_hl = (r_bg >= r_bg_med) & (g_bg < g_bg_med) & roi_mask_full
    mask_ll = (r_bg < r_bg_med)  & (g_bg < g_bg_med) & roi_mask_full
    
    return r_bg, g_bg, mask_hh, mask_lh, mask_hl, mask_ll

def save_masks_to_tifs(zip_path, mask_hh, mask_lh, mask_hl, mask_ll):
    """Saves the 4 boolean masks as 4 separate 8-bit TIF files."""
    base_path = Path(zip_path)
    output_dir = base_path.parent
    base_name = base_path.stem  # Removes '.zip' if that was originally passed
    
    for suffix, mask in [('HH', mask_hh), ('LH', mask_lh), ('HL', mask_hl), ('LL', mask_ll)]:
        # Skip completely empty masks
        if not np.any(mask):
            continue
            
        # Convert boolean mask to 8-bit grayscale (0 and 255)
        mask_8bit = (mask * 255).astype(np.uint8)
        
        # Construct the output file path (e.g., "rois_HH.tif")
        output_file = output_dir / f"{base_name}_{suffix}.tif"
        
        # Save as a standard TIF file
        tifffile.imwrite(output_file, mask_8bit)

if __name__ == "__main__":
    # Allows running independently to only extract the regions
    
    user_path = cfg.USER_PATH
    print(f"Running independent region separation on: {user_path}")
    
    img_r = tifffile.imread(cfg.R_PATH)
    img_g = tifffile.imread(cfg.G_PATH)      
    
    roi_zip_path = user_path / "rois.zip"
    if not roi_zip_path.exists():
        print(f"Error: rois.zip not found in {user_path}")
    else:
        print("Reading ROIs...")
        rois = read_roi_zip(roi_zip_path)        
        for roi_name, roi_data in rois.items():
            print(f"Processing ROI: {roi_name}")
            roi_mask_full = roi_to_mask(roi_data, img_r.shape)
            
            r_bg, g_bg, mask_hh, mask_lh, mask_hl, mask_ll = separate_regions_and_bg(img_r, img_g, roi_mask_full)
            
            zip_filename = user_path / f"{cfg.R_RAW.stem}_{roi_name}_regions.zip"
            save_masks_to_tifs(zip_filename, mask_hh, mask_lh, mask_hl, mask_ll)
            print(f"Saved: {zip_filename.name}")