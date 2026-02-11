import os
import zipfile
import numpy as np
import tifffile
from pathlib import Path
from itertools import combinations_with_replacement
from scipy import ndimage as ndi
from skimage import draw
from skimage.morphology import remove_small_objects
from read_roi import read_roi_zip  # pip install read-roi

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

def filament_3d_wrapper(struct_img: np.ndarray, f3_param):
    """Wrapper for 3D filament filter."""
    bw = np.zeros(struct_img.shape, dtype=bool)
    for fid in range(len(f3_param)):
        sigma = f3_param[fid][0]
        # Calculate Hessian
        eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
        # Calculate Vesselness
        responce = compute_vesselness3D(eigenvalues[1], eigenvalues[2], tau=1)
        # Apply Cutoff and OR operation
        bw = np.logical_or(bw, responce > f3_param[fid][1])
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

def get_roi_bounding_box(roi_data):
    """Extract bounding box (y_min, y_max, x_min, x_max) from ROI data."""
    # ImageJ ROIs usually provide top, left, width, height or coordinates
    if 'y' in roi_data and 'x' in roi_data:
        # Rectangle or similar
        r_top = roi_data['y']
        r_left = roi_data['x']
        r_height = roi_data['height']
        r_width = roi_data['width']
        return int(r_top), int(r_top + r_height), int(r_left), int(r_left + r_width)
    
    # Fallback for polygon/freehand types if they store explicit coords
    if 'x' in roi_data and isinstance(roi_data['x'], list):
        xs = roi_data['x']
        ys = roi_data['y']
        return int(min(ys)), int(max(ys)), int(min(xs)), int(max(xs))
        
    raise ValueError("Could not determine bounding box for ROI")

def roi_to_mask(roi, image_shape):
    """
    Convert ROI data to a binary mask using specific logic for different shapes.
    Adapted from segmentation_pfkl_2d.py
    """
    mask = np.zeros(image_shape, dtype=bool)
    # Get type safely, default to unknown
    roi_type = roi.get('type', '').lower() 
    
    try:
        if roi_type == 'rectangle':
            # Rectangles use top/left/width/height
            r_start = int(roi['top'])
            r_end = int(roi['top'] + roi['height'])
            c_start = int(roi['left'])
            c_end = int(roi['left'] + roi['width'])
            
            # Clip to image bounds to avoid errors
            r_start = max(0, r_start)
            c_start = max(0, c_start)
            r_end = min(image_shape[0], r_end)
            c_end = min(image_shape[1], c_end)
            
            mask[r_start:r_end, c_start:c_end] = True
            
        elif roi_type in ['polygon', 'freehand', 'traced']:
            # Polygons use lists of x and y coordinates
            r = roi['y']
            c = roi['x']
            rr, cc = draw.polygon(r, c, shape=image_shape)
            mask[rr, cc] = True
            
        elif roi_type == 'oval':
            r_center = roi['top'] + roi['height'] / 2
            c_center = roi['left'] + roi['width'] / 2
            r_radius = roi['height'] / 2
            c_radius = roi['width'] / 2
            rr, cc = draw.ellipse(r_center, c_center, r_radius, c_radius, shape=image_shape)
            mask[rr, cc] = True
            
    except Exception as e:
        print(f"    Warning: ROI conversion failed for type '{roi_type}': {e}")
        
    return mask

def process_single_image(image_path, sigma_cutoff_pairs):
    print(f"Processing: {image_path}")
    image_path = Path(image_path)
    
    # 1. Load Image
    try:
        img = tifffile.imread(image_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # Check dims
    ndim = img.ndim
    is_3d = ndim == 3
    print(f"Image shape: {img.shape}, Detected as {'3D' if is_3d else '2D'}")

    # 2. Search for ROIs (Try specific name first, then generic rois.zip)
    potential_paths = [
        image_path.parent / f"{image_path.stem}_rois.zip",
        image_path.parent / "rois.zip"
    ]
    
    rois = None
    for p in potential_paths:
        if p.exists():
            try:
                rois = read_roi_zip(p)
                print(f"  Found ROIs at: {p.name}")
                break
            except Exception as e:
                print(f"  Error reading {p.name}: {e}")

    if not rois:
        print(f"No ROI file found. Skipping.")
        return

    # 3. Iterate ROIs
    for i, (roi_name, roi_data) in enumerate(rois.items()):
        roi_idx = i + 1
        print(f"  > Processing ROI {roi_idx}: {roi_name}")
        
        # === CHANGED PART START ===
        # Instead of 'get_roi_bounding_box', we generate the mask first (Robust Method)
        
        # Calculate mask for the specific 2D plane
        shape_2d = img.shape[-2:] # Height, Width
        roi_mask = roi_to_mask(roi_data, shape_2d)
        
        # Get Bounding Box from the binary mask
        coords = np.argwhere(roi_mask)
        if coords.size == 0:
            print("    ROI mask is empty (possibly outside image bounds). Skipping.")
            continue
            
        # Calculate bounds (min/max of True pixels)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0) + 1 # +1 because slices are exclusive
        
        # Crop Image
        if is_3d:
            img_crop = img[:, y1:y2, x1:x2]
        else:
            img_crop = img[y1:y2, x1:x2]
            
        # Crop the ROI mask as well (to apply after segmentation)
        roi_mask_crop = roi_mask[y1:y2, x1:x2]
        # === CHANGED PART END ===

        if img_crop.size == 0:
            print("    ROI crop is empty. Skipping.")
            continue

        # 4. Normalization (Sample 1/1000 pixels)
        flat_sample = img_crop.ravel()[::1000] 
        if flat_sample.size == 0:
            flat_sample = img_crop.ravel()
            
        val_min = float(flat_sample.min())
        val_max = float(flat_sample.max())
        
        img_norm = (img_crop.astype(float) - val_min) / (val_max - val_min + 1e-10)
        img_norm = np.clip(img_norm, 0, 1)

        # 5. Segmentation
        print("    Running segmentation...")
        if is_3d:
            seg_crop = filament_3d_wrapper(img_norm, sigma_cutoff_pairs)
        else:
            seg_crop = filament_2d_wrapper(img_norm, sigma_cutoff_pairs)

        # 6. Apply ROI Polygon Mask (Pixels outside the exact shape become 0)
        if is_3d:
            seg_crop = np.logical_and(seg_crop, roi_mask_crop[np.newaxis, :, :])
        else:
            seg_crop = np.logical_and(seg_crop, roi_mask_crop)

        # 7. Cleanup
        seg_crop = remove_small_objects(seg_crop, min_size=16)

        # 8. Output Construction
        full_mask = np.zeros(img.shape, dtype=np.uint8)
        
        if is_3d:
            full_mask[:, y1:y2, x1:x2] = seg_crop.astype(np.uint8) * 255
        else:
            full_mask[y1:y2, x1:x2] = seg_crop.astype(np.uint8) * 255

        # Save
        output_name = f"{image_path.stem}_{roi_idx}_mask.tif"
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
        [1.0, 0.1],  
        [1.5, 0.2]
    ]
    
    # Path to your image
    TARGET_IMAGE = r"F:\20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM\Cell6\1_AddSalt45min\Cell0208.tif" 

    # --- RUN ---
    process_single_image(TARGET_IMAGE, FILAMENT_PARAMS)