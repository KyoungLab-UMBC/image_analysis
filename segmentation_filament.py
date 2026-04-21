import numpy as np
import tifffile
from pathlib import Path
from itertools import combinations_with_replacement
from scipy import ndimage as ndi
from skimage import draw
from skimage.morphology import remove_small_objects
from read_roi import read_roi_zip
import util.config as cfg
import util.roi_to_mask as roi_tools
import util.background_correction as bg_tools
from util.normalization import normalize_minmax
# ==========================================
# PART 1: Core Algorithm Functions (Merged)
# ==========================================

def divide_nonzero(array1, array2):
    """Divides two arrays. Returns zero when dividing by zero."""
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def sortbyabs(a: np.ndarray, axis=0):
    """Sort array along a given axis by the absolute value"""
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[tuple(index)]

def absolute_eigenvaluesh(nd_array):
    """Computes the eigenvalues sorted by absolute value."""
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    return [
        np.squeeze(eigenvalue, axis=-1)
        for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)
    ]

def compute_3d_hessian_matrix(nd_array: np.ndarray, sigma: float = 1, scale: bool = True, whiteonblack: bool = True) -> np.ndarray:
    """Computes the hessian matrix for an nd_array."""
    ndim = nd_array.ndim
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma, mode="nearest", truncate=3.0)
    gradient_list = np.gradient(smoothed)
    hessian_elements = [
        np.gradient(gradient_list[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(range(ndim), 2)
    ]

    if sigma > 0 and scale:
        if whiteonblack:
            hessian_elements = [(sigma**2) * element for element in hessian_elements]
        else:
            hessian_elements = [-1 * (sigma**2) * element for element in hessian_elements]

    hessian_full = [[()] * ndim for x in range(ndim)]
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian_full[ax0][ax1] = element
        if ax0 != ax1:
            hessian_full[ax1][ax0] = element

    hessian_rows = list()
    for row in hessian_full:
        hessian_rows.append(np.stack(row, axis=-1))

    return np.stack(hessian_rows, axis=-2)

def absolute_3d_hessian_eigenvalues(nd_array, sigma=1, scale=True, whiteonblack=True):
    """Eigenvalues of the hessian matrix sorted by absolute value."""
    return absolute_eigenvaluesh(
        compute_3d_hessian_matrix(nd_array, sigma=sigma, scale=scale, whiteonblack=whiteonblack)
    )

def compute_vesselness3D(eigen2, eigen3, tau):
    """Backend for computing 3D filament filter."""
    lambda3m = np.copy(eigen3)
    lambda3m[np.logical_and(eigen3 < 0, eigen3 > (tau * eigen3.min()))] = tau * eigen3.min()
    response = np.multiply(np.square(eigen2), np.abs(lambda3m - eigen2))
    response = divide_nonzero(27 * response, np.power(2 * np.abs(eigen2) + np.abs(lambda3m - eigen2), 3))
    response[np.less(eigen2, 0.5 * lambda3m)] = 1
    response[eigen2 >= 0] = 0
    response[eigen3 >= 0] = 0
    response[np.isinf(response)] = 0
    return response

def compute_vesselness2D(eigen2, tau):
    """Backend for computing 2D filament filter."""
    Lambda3 = np.copy(eigen2)
    Lambda3[np.logical_and(Lambda3 < 0, Lambda3 >= (tau * Lambda3.min()))] = tau * Lambda3.min()
    response = np.multiply(np.square(eigen2), np.abs(Lambda3 - eigen2))
    response = divide_nonzero(27 * response, np.power(2 * np.abs(eigen2) + np.abs(Lambda3 - eigen2), 3))
    response[np.less(eigen2, 0.5 * Lambda3)] = 1
    response[eigen2 >= 0] = 0
    response[np.isinf(response)] = 0
    return response

def vesselnessSliceBySlice(struct_img: np.ndarray, f_param: list):
    """wrapper for slice-by-slice 2d filament filter on a 3D stack"""
    assert len(struct_img.shape) == 3, "image has to be 3D"
    bw = np.zeros(struct_img.shape, dtype=bool)
    
    for fid in range(len(f_param)):
        sigma = f_param[fid][0]
        cutoff = f_param[fid][1]
        
        for zz in range(struct_img.shape[0]):
            slice_img = struct_img[zz, :, :]
            
            # Compute Hessian eigenvalues for the 2D slice
            # absolute_3d_hessian_eigenvalues supports N-dimensional arrays
            eigenvalues = absolute_3d_hessian_eigenvalues(slice_img, sigma=sigma, scale=True, whiteonblack=True)
            
            # Compute 2D vesselness (using the largest absolute eigenvalue)
            response = compute_vesselness2D(eigenvalues[1], tau=1)
            
            # Apply cutoff and combine with the boolean mask using logical OR
            bw[zz, :, :] = np.logical_or(bw[zz, :, :], response > cutoff)
            
    return bw

def filament_2d_wrapper(struct_img: np.ndarray, f2_param):
    """Wrapper for 2D filament filter (handles 3D stack slice-by-slice with MIP)."""
    bw = np.zeros(struct_img.shape, dtype=bool)

    if len(struct_img.shape) == 2:
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
            eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
            responce = compute_vesselness2D(eigenvalues[1], tau=1)
            bw = np.logical_or(bw, responce > f2_param[fid][1])
            
    elif len(struct_img.shape) == 3:
        # Specialized 3D processing for 2D filaments (MIP concatenation trick)
        mip = np.amax(struct_img, axis=0)
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
            res = np.zeros_like(struct_img)
            for zz in range(struct_img.shape[0]):
                tmp = np.concatenate((struct_img[zz, :, :], mip), axis=1)
                eigenvalues = absolute_3d_hessian_eigenvalues(tmp, sigma=sigma, scale=True, whiteonblack=True)
                responce = compute_vesselness2D(eigenvalues[1], tau=1)
                res[zz, :, : struct_img.shape[2] - 3] = responce[:, : struct_img.shape[2] - 3]
            bw = np.logical_or(bw, res > f2_param[fid][1])
    return bw

# ==========================================
# PART 2: Main Processing Pipeline
# ==========================================

def process_single_image(image_path, sigma_cutoff_pairs):
    print(f"Processing: {image_path}")
    image_path = Path(image_path)
    
    # 1. Load Image
    try:
        img = tifffile.imread(image_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        return
    
    img = ndi.gaussian_filter(img, sigma=1.0)

    # Check dims
    ndim = img.ndim
    is_3d = ndim == 3
    print(f"Image shape: {img.shape}, Detected as {'3D' if is_3d else '2D'}")

    if is_3d:
        # ==========================================
        # 3D PROCESSING PIPELINE
        # ==========================================
        cell_mask_path = cfg.CELL_MASK
        roi_name = cell_mask_path.stem
        print(f"  > Processing 3D volume using mask: {roi_name}")

        try:
            roi_mask = tifffile.imread(cell_mask_path).astype(bool)
        except Exception as e:
            print(f"    Error reading 3D cell mask {cell_mask_path}: {e}")
            return
            
        # Get Bounding Box from the binary mask
        coords = np.argwhere(roi_mask)
        if coords.size == 0:
            print("    ROI mask is empty. Skipping.")
            return
            
        # Calculate bounds (min/max of True pixels) for 3D: Z, Y, X
        z1, y1, x1 = coords.min(axis=0)
        z2, y2, x2 = coords.max(axis=0) + 1 # +1 because slices are exclusive
        
        # Crop the image in all 3 dimensions
        img_crop_raw = img[z1:z2, y1:y2, x1:x2]
        
        # Crop the ROI mask as well (to apply after segmentation)
        roi_mask_crop = roi_mask[z1:z2, y1:y2, x1:x2]

        img_norm = normalize_minmax(img_crop_raw, high_bright=True)

        # Segmentation
        print("    Running 3D segmentation...")
        seg_crop = vesselnessSliceBySlice(img_norm, sigma_cutoff_pairs)

        seg_crop = np.logical_and(seg_crop, roi_mask_crop)

        # Cleanup
        seg_crop = remove_small_objects(seg_crop, min_size=16)
        # 8. Output Construction
        full_mask = np.zeros(img.shape, dtype=np.uint8)
        full_mask[z1:z2, y1:y2, x1:x2] = seg_crop.astype(np.uint8) * 255

    else:
        # ==========================================
        # 2D PROCESSING PIPELINE
        # ==========================================
        # Search for ROIs (Try specific name first, then generic rois.zip)
        rois = roi_tools.load_roi_file(cfg.R_RAW)

        # Iterate ROIs
        for roi_name, roi_data in rois.items():
            print(f"  > Processing ROI: {roi_name}")
            
            # Calculate mask for the specific 2D plane
            shape_2d = img.shape[-2:] # Height, Width
            roi_mask = roi_tools.roi_to_mask(roi_data, shape_2d)
            
            # Get Bounding Box from the binary mask
            coords = np.argwhere(roi_mask)
            if coords.size == 0:
                print("    ROI mask is empty (possibly outside image bounds). Skipping.")
                continue
                
            # Calculate bounds (min/max of True pixels)
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0) + 1 # +1 because slices are exclusive
            
            # Crop Image
            img_crop_raw = img[y1:y2, x1:x2]
                
            # Crop the ROI mask as well (to apply after segmentation)
            roi_mask_crop = roi_mask[y1:y2, x1:x2]
            # === CHANGED PART END ===

            if img_crop_raw.size == 0:
                print("    ROI crop is empty. Skipping.")
                continue
            
            # 4. Normalization (Sample 1/100 pixels)
            img_crop = bg_tools.estimate_background_rolling_ball(img_crop_raw, radius=10, create_background=False, use_paraboloid=False)
            img_norm = normalize_minmax(img_crop)

            # Segmentation
            print("    Running 2D segmentation...")
            seg_crop = filament_2d_wrapper(img_norm, sigma_cutoff_pairs)

            # Apply ROI Polygon Mask
            seg_crop = np.logical_and(seg_crop, roi_mask_crop)

            # Cleanup
            seg_crop = remove_small_objects(seg_crop, min_size=16)

            # Output Construction
            full_mask = np.zeros(img.shape, dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = seg_crop.astype(np.uint8) * 255

    # Save
    output_name = f"{image_path.stem}_{roi_name}_mask.tif"
    output_path = image_path.parent / output_name
    tifffile.imwrite(output_path, full_mask)
    print(f"    Saved: {output_name}")

# ==========================================
# PART 3: Execution Configuration
# ==========================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Define your parameters here
    # Format: [[scale, cutoff], [scale, cutoff], ...]
    # Adjust these based on your specific filament thickness and brightness
    FILAMENT_PARAMS = [
        [1.0, 0.20],  
        [1.5, 0.20]
    ]
    bg_tools.init_imagej()  # Ensure ImageJ is initialized before processing
    # --- RUN ---
    process_single_image(cfg.MITO_RAW, FILAMENT_PARAMS)