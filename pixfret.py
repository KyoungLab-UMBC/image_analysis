import logging
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import roifile
from util.roi_to_mask import roi_to_mask, load_roi_file
import util.config as config

# Configure basic logging for a cleaner output than print statements
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Mathematical Models for Spectral Bleed-Through (SBT) ---
# Based on the original PixFRET article equations (Figure 3)
def model_constant(x, a):
    return np.full_like(x, a, dtype=float)

def model_linear(x, a, b):
    return a * x + b

def model_exponential(x, a, b, c):
    # Prevent overflow by capping the exponent max value (700 is just below float64 limit)
    safe_exponent = np.clip(c * x, -700.0, 700.0)
    return a + b * np.exp(safe_exponent)

SBT_MODELS = {
    'constant': (model_constant, [0.1]),
    'linear': (model_linear, [0.0, 0.1]),
    # Changed initial guess for 'b' from 0.01 to 1e-5 to scale properly with 16-bit pixel intensities
    'expo': (model_exponential, [0.0, 0.1, 1e-5]) 
}

def calculate_background(img: np.ndarray):
    img = img.astype(np.float32)
    
    # Sample 1 out of every 4096 pixels
    sample = img.ravel()[::4096]
    
    # Fallback: If image is smaller than 4096 pixels, use the whole image
    if sample.size == 0:
        sample = img
    
    # Get the minimum 10% of the sampled pixels
    n_min = max(1, int(sample.size * 0.1))
    
    # np.partition is highly optimized for finding the top/bottom k elements
    bottom_10_percent = np.partition(sample, n_min)[:n_min]
    return np.mean(bottom_10_percent)

def load_and_preprocess_cell(folder: Path, fret_time: float, donor_time: float, acceptor_time: float):
    """
    Reads the TIFF files, applies time correction, and calculates backgrounds.
    Assumes ascending sort order: [0] = Acceptor, [1] = Donor, [2] = FRET.
    """
    # Find and sort 4-digit tif files
    tif_files = sorted(folder.glob("Cell[0-9][0-9][0-9][0-9].tif"))
    if len(tif_files) < 3:
        logging.warning(f"Skipping {folder.name} - Not enough TIFF files.")
        return None

    # Load raw images
    acc_raw = gaussian_filter(tifffile.imread(tif_files[0]).astype(float), sigma=1, mode="nearest", truncate=3.0)
    don_raw = gaussian_filter(tifffile.imread(tif_files[1]).astype(float), sigma=1, mode="nearest", truncate=3.0)
    fret_raw = gaussian_filter(tifffile.imread(tif_files[2]).astype(float), sigma=1, mode="nearest", truncate=3.0)

    # Calculate backgrounds
    acc_bg = calculate_background(acc_raw)
    don_bg = calculate_background(don_raw)
    fret_bg = calculate_background(fret_raw)

    print(f"Backgrounds for {folder.name} - Acc: {acc_bg:.2f}, Don: {don_bg:.2f}, FRET: {fret_bg:.2f}")

    # Threshold masks (PixFRET user guide: calculate only if pixels are > background)
    valid_mask = (acc_raw > 2 * acc_bg) & (don_raw > 2 * don_bg) & (fret_raw > 2 * fret_bg)

    # Background subtraction
    acc_sub = np.maximum(acc_raw - acc_bg, 0)
    don_sub = np.maximum(don_raw - don_bg, 0)
    fret_sub = np.maximum(fret_raw - fret_bg, 0)

    # Exposure time corrections (relative to FRET time)
    acc_corr = acc_sub * (fret_time / acceptor_time)
    don_corr = don_sub * (fret_time / donor_time)

    return {
        'acc': acc_corr, 
        'don': don_corr, 
        'fret': fret_sub, 
        'mask': valid_mask,
        'shape': acc_raw.shape
    }

def get_roi_mask(folder: Path, shape: tuple) -> np.ndarray:
    """
    Reads rois.zip and returns a unified boolean mask for the regions.
    """
    roi_path = folder / "rois.zip"
    mask = np.zeros(shape, dtype=bool)
    
    if not roi_path.exists():
        return mask

    try:
        from skimage.draw import polygon
        rois = roifile.ImagejRoi.fromfile(roi_path)
        rois = [rois] if not isinstance(rois, list) else rois
        
        for roi in rois:
            coords = roi.coordinates()
            if coords is not None:
                # Clip coords to image bounds
                c = np.clip(coords[:, 0], 0, shape[1] - 1)
                r = np.clip(coords[:, 1], 0, shape[0] - 1)
                rr, cc = polygon(r, c, shape)
                mask[rr, cc] = True
    except Exception as e:
        logging.error(f"Error processing ROI in {folder.name}: {e}")
        
    return mask

def process_sbt_mode(base_path: Path, mode: str, fret_time: float, donor_time: float, acceptor_time: float):
    """
    Aggregates ROI pixels across all cells in the path and fits the SBT models.
    """
    logging.info(f"--- Starting {mode.upper()} Mode Analysis ---")
    x_pixels, y_pixels = [], []

    for folder in base_path.glob("Cell*"):
        if not folder.is_dir():
            continue

        imgs = load_and_preprocess_cell(folder, fret_time, donor_time, acceptor_time)
        if not imgs:
            continue

        # Get the path to the first TIFF to use with your load_roi_file logic
        reference_tif = next(folder.glob("Cell[0-9][0-9][0-9][0-9].tif"), None)
        if not reference_tif:
            continue

        # Load ROIs using your util function (returns a dictionary of ROIs)
        rois = load_roi_file(reference_tif)
        if not rois:
            continue

        # For SBT, x is the main channel intensity, y is the FRET channel bleed-through
        x_channel = imgs['don'] if mode.upper() == "DONOR_SBT" else imgs['acc']
        fret_channel = imgs['fret']

        # Process EACH ROI individually to calculate its specific median
        for roi_name, roi_data in rois.items():
            roi_mask = roi_to_mask(roi_data, imgs['shape'])
            combined_mask = roi_mask & imgs['mask']
            
            if not np.any(combined_mask):
                continue
                
            # Extract FRET pixels for this specific ROI to find the median
            x_roi_pixels = x_channel[combined_mask]
            roi_x_threshold = np.percentile(x_roi_pixels, 95)
            fret_roi_pixels = fret_channel[combined_mask]
            roi_fret_threshold = np.percentile(fret_roi_pixels, 98)
            
            # Discard lowest 50% pixels: only keep those STRICTLY ABOVE the median
            median_filtered_mask = combined_mask & (x_channel >= roi_x_threshold) & (fret_channel <= roi_fret_threshold)
            
            if not np.any(median_filtered_mask):
                continue

            # Append the filtered pixels
            x_pixels.extend(x_channel[median_filtered_mask])
            y_pixels.extend(fret_channel[median_filtered_mask])

    if not x_pixels:
        logging.error("No valid ROI pixels found across the dataset. Cannot compute SBT.")
        return

    x_arr = np.array(x_pixels)
    y_arr = np.array(y_pixels)
    sbt_ratio = y_arr / x_arr

    # Fit all models and print R^2
    for model_name, (func, p0) in SBT_MODELS.items():
        try:
            popt, _ = curve_fit(func, x_arr, sbt_ratio, p0=p0, maxfev=10000)
            y_pred = func(x_arr, *popt)
            
            # R-squared calculation
            ss_res = np.sum((sbt_ratio - y_pred) ** 2)
            ss_tot = np.sum((sbt_ratio - np.mean(sbt_ratio)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            logging.info(f"{model_name.capitalize()} Fit -> Params: {np.round(popt, 6)} | R^2 = {r2:.4f}")
        except Exception as e:
            logging.error(f"{model_name.capitalize()} Fit failed: {e}")

def process_fret_mode(base_path: Path, fret_time: float, donor_time: float, acceptor_time: float, 
                      donor_model: str, donor_params: list, acceptor_model: str, acceptor_params: list):
    """
    Computes normalized FRET for all cell folders using user-defined SBT parameters.
    """
    logging.info("--- Starting FRET Computation Mode ---")
    
    don_func = SBT_MODELS[donor_model.lower()][0]
    acc_func = SBT_MODELS[acceptor_model.lower()][0]

    import concurrent.futures

    def process_folder(folder: Path):
        imgs = load_and_preprocess_cell(folder, fret_time, donor_time, acceptor_time)
        if not imgs:
            return

        don, acc, fret, valid_mask = imgs['don'], imgs['acc'], imgs['fret'], imgs['mask']

        # Calculate BT using chosen models
        don_sbt = don_func(don, *donor_params)
        acc_sbt = acc_func(acc, *acceptor_params)

        # PixFRET formula (from the article)
        fret_corrected = fret - (don * don_sbt) - (acc * acc_sbt)

        # Format outputs: zero out invalid pixels
        fret_corrected[~valid_mask] = 0

        # Denominator: square root of product of donor and acceptor intensities
        denom = np.sqrt(don * acc)
        
        # Initialize NFRET array
        nfret = np.zeros_like(fret_corrected)
        
        # Compute NFRET only where pixels exceed background thresholds and denom > 0
        calc_mask = valid_mask & (denom > 0)
        nfret[calc_mask] = (fret_corrected[calc_mask] / denom[calc_mask]) * 1000.0
        
        # Convert to 16-bit TIFF
        fret_16bit = np.clip(np.nan_to_num(fret_corrected), 0, 65535).astype(np.uint16)
        nfret_16bit = np.clip(np.nan_to_num(nfret), 0, 65535).astype(np.uint16)

        # Save outputs
        tifffile.imwrite(folder / "FRET_corrected.tif", fret_16bit)
        tifffile.imwrite(folder / "NFRET(x1000).tif", nfret_16bit)
        logging.info(f"Saved FRET and NFRET images to {folder}")

    folders_to_process = []
    for cell_folder in base_path.glob("Cell*"):
        if not cell_folder.is_dir():
            continue
        # Process the cell_folder itself if it contains TIFFs
        folders_to_process.append(cell_folder)
        # Add all subdirectories inside cell_folder
        folders_to_process += [f for f in cell_folder.iterdir() if f.is_dir()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_folder, folders_to_process)

# ==========================================
# Execution Configuration
# ==========================================
if __name__ == "__main__":
    # DATA_PATH = Path(r"F:\20260507 PFKL-mCherry_HKII-EGFP_MitotrackerDeepred + FRET correction - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM")
    DATA_PATH = config.ROOT_PATH

    # Exposure settings (ms)
    FRET_T = 200.0
    DON_T = 100.0
    ACC_T = 100.0

    # Choose Mode: "FRET", "DONOR_SBT", or "ACCEPTOR_SBT"
    RUN_MODE = "FRET"  

    if RUN_MODE in ["DONOR_SBT", "ACCEPTOR_SBT"]:
        process_sbt_mode(DATA_PATH, RUN_MODE, FRET_T, DON_T, ACC_T)
        
    elif RUN_MODE == "FRET":
        # Modify these with the values obtained from running the SBT modes above
        process_fret_mode(
            base_path=DATA_PATH, 
            fret_time=FRET_T, donor_time=DON_T, acceptor_time=ACC_T,
            donor_model="constant", 
            donor_params=[0.056004],         # E.g. extracted parameter 'a'
            acceptor_model="constant", 
            acceptor_params=[0.070145] # E.g. extracted parameters 'a', 'b', 'c'
        )