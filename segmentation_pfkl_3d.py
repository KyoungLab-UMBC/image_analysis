from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology, draw, segmentation, feature
from read_roi import read_roi_zip
import util.background_correction as bg_tools
import util.config as cfg
import util.roi_to_mask as roi_tools
from util.normalization import normalize_minmax

def dot_3d(struct_img, log_sigma, cutoff=-1):
    """Apply 3D spot filter (LoG)"""
    if struct_img.ndim != 3:
        print(f"Warning: dot_3d received {struct_img.ndim}D image")
    
    # LoG response
    response = -1 * (log_sigma**2) * ndi.gaussian_laplace(struct_img, log_sigma)
    
    if cutoff < 0:
        return response
    else:
        return response > cutoff


def filter_shape(mask, max_ar=2.6, min_volume=5, min_solidity=0.85):
    """
    Filters 3D objects by:
    1. Min Volume (removes tiny noise < 5 voxels)
    2. Aspect Ratio (removes lines)
    3. Solidity (removes jagged/worm shapes)
    """
    labeled_img = measure.label(mask)
    regions = measure.regionprops(labeled_img)
    new_mask = np.zeros_like(mask, dtype=bool)
    
    for region in regions:
        # 1. Min Volume Filter
        if region.area < min_volume: continue
        
        # 3. Aspect Ratio Filter
        if region.minor_axis_length == 0:
            ar = 9999.0
        else:
            ar = region.major_axis_length / region.minor_axis_length
            
        if ar > max_ar: continue

        # 4. Solidity Filter (Keeps compact dots, removes jagged lines)
        if region.solidity < min_solidity: continue
            
        coords = region.coords
        # 3D indexing: Z, Y, X
        new_mask[coords[:,0], coords[:,1], coords[:,2]] = True
        
    return new_mask

def _remove_diag_2d(binary_mask_2d):
    """Helper function to break 2D diagonal bridges"""
    TL = binary_mask_2d[:-1, :-1]
    TR = binary_mask_2d[:-1, 1:]
    BL = binary_mask_2d[1:, :-1]
    BR = binary_mask_2d[1:, 1:]
    
    mask_a = (TL == True) & (TR == False) & (BL == False) & (BR == True)
    BR[mask_a] = False
    
    mask_b = (TL == False) & (TR == True) & (BL == True) & (BR == False)
    BL[mask_b] = False
    
    return binary_mask_2d

def remove_diagonal_bridges_3d(binary_mask):
    """
    Detects diagonal connections in 3D and breaks them.
    We apply the 2D logic across all three orthogonal planes (XY, XZ, YZ).
    """
    mask_out = binary_mask.copy()
    
    # 1. Break bridges in the XY planes (Iterate over Z)
    for z in range(mask_out.shape[0]):
        mask_out[z, :, :] = _remove_diag_2d(mask_out[z, :, :])
        
    # 2. Break bridges in the YZ planes (Iterate over X)
    for x in range(mask_out.shape[2]):
        mask_out[:, :, x] = _remove_diag_2d(mask_out[:, :, x])
        
    # 3. Break bridges in the XZ planes (Iterate over Y)
    for y in range(mask_out.shape[1]):
        mask_out[:, y, :] = _remove_diag_2d(mask_out[:, y, :])
        
    return mask_out

def separate_attached_objects_watershed(binary_mask, min_distance):
    """
    Separates attached 3D objects in a binary mask using Watershed.
    Returns a binary mask with lines separating the objects.
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
        
    # 1. Distance Transform (N-D aware)
    distance = ndi.distance_transform_edt(binary_mask)
    
    # 2. Find Peaks (Markers)
    coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    
    markers, _ = ndi.label(mask_peaks)
    
    # 3. Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary_mask, watershed_line=True)
    
    # 4. Convert back to binary (labels > 0)
    return labels > 0

def smart_merge_objects(mask_large, mask_small):
    """
    Merges mask_large and mask_small using watershed based on specific overlap rules.
    """
    # 1. Merge mask_large and mask_small as union_mask
    union_mask = mask_large | mask_small
    if np.sum(union_mask) == 0:
        return union_mask
        
    # Label all components to identify individual objects globally
    # Use connectivity=3 for full 3D (26-connectivity)
    labeled_union = measure.label(union_mask, connectivity=3)
    labeled_small = measure.label(mask_small, connectivity=3)
    labeled_large = measure.label(mask_large, connectivity=3)
    
    # Initialize the marker array
    markers = np.zeros_like(union_mask, dtype=np.int32)
    current_label = 1
    
    # 2. For each object in the union_mask
    for region in measure.regionprops(labeled_union):
        r_slice = region.slice
        union_obj = region.image
        
        local_mask_large = mask_large[r_slice]
        local_mask_small = mask_small[r_slice]
        
        local_small_labels = labeled_small[r_slice].copy()
        local_small_labels[~union_obj] = 0
        
        local_large_labels = labeled_large[r_slice].copy()
        local_large_labels[~union_obj] = 0
        
        area_union = region.area
        markers_in_this_union = 0
        
        m_slice = markers[r_slice]
        
        # --- For each object in mask_small ---
        small_ids = np.unique(local_small_labels)
        small_ids = small_ids[small_ids > 0]
        
        for sid in small_ids:
            small_obj = (local_small_labels == sid)
            area_small = np.sum(small_obj)
            area_overlap = np.sum(small_obj & local_mask_large)
            
            if area_small > 0:
                ratio_overlap = area_overlap / area_small
                ratio_union = area_small / area_union
                
                if ratio_union >= 0.1:
                    m_slice[small_obj] = current_label
                    current_label += 1
                    markers_in_this_union += 1
                    
        # --- For each object in mask_large ---
        large_ids = np.unique(local_large_labels)
        large_ids = large_ids[large_ids > 0]
        
        for lid in large_ids:
            large_obj = (local_large_labels == lid)
            area_large = np.sum(large_obj)
            
            overlap_mask = large_obj & local_mask_small
            area_overlap = np.sum(overlap_mask)
            
            if area_large > 0:
                ratio_overlap = area_overlap / area_large
                ratio_large_union = area_large / area_union
                
                if ratio_overlap <= 0.15 and ratio_large_union <= 0.6:
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                elif ratio_large_union <= 0.25:
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                elif ratio_overlap <= 0.15 and ratio_large_union > 0.91:
                    overlapping_marker_ids = np.unique(m_slice[overlap_mask])                    
                    for m_id in overlapping_marker_ids:
                        if m_id > 0:
                            m_slice[m_slice == m_id] = 0
                            
                if not np.any(large_obj & (m_slice > 0)):
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                
        if markers_in_this_union == 0:
            free_pixels = union_obj & (m_slice == 0)
            if np.any(free_pixels):
                m_slice[free_pixels] = current_label
                current_label += 1
                
    # 3. Do the watershed with the markers above
    distance = ndi.distance_transform_edt(union_mask)
    
    shrunk_markers = np.zeros_like(markers)
    for marker_id in np.unique(markers)[1:]:
        marker_mask = (markers == marker_id)
        local_dist = distance * marker_mask
        
        max_idx = np.argmax(local_dist)
        coords = np.unravel_index(max_idx, local_dist.shape)
        
        shrunk_markers[coords] = marker_id
        
    labels = segmentation.watershed(-distance, shrunk_markers, mask=union_mask, watershed_line=True)
    
    return labels > 0

def get_mask(img, log_sigma, cutoff, min_distance):
    """Helper function to apply dot_3d and basic cleaning in one step"""
    mask_1 = dot_3d(img, log_sigma, cutoff)
    mask_2 = dot_3d(img, log_sigma/2, cutoff/2)
    mask_max = (img == 1.0)

    merged = mask_1 | mask_2 | mask_max
    mask = remove_diagonal_bridges_3d(separate_attached_objects_watershed(merged, min_distance))
    return mask

def bg_subtraction_3d(img_3d, radius):
    """Applies rolling ball background subtraction slice-by-slice to a 3D image."""
    res = np.zeros_like(img_3d, dtype=np.float32)
    for z in range(img_3d.shape[0]):
        # Apply 2D background subtraction on each Z slice
        res[z] = bg_tools.estimate_background_rolling_ball(img_3d[z], radius=radius, create_background=False)
    return res

def process_roi_image(roi_img, roi_mask, save_prefix=None):
    """
    Applies the 2-branch logic to a single cropped 3D ROI image.
    """
    # === BRANCH 1: Radius 20 (Large/Medium features) ===
    # Using the 3D wrapper to do slice-by-slice rolling ball
    sub_1 = bg_subtraction_3d(roi_img, radius=10)
    
    sub_1 = np.clip(sub_1, 0, None)
    
    norm_1_b = normalize_minmax(sub_1, high_bright=True)
    norm_1_d = normalize_minmax(sub_1, high_bright=False)
    
    mask_1_b = get_mask(norm_1_b, log_sigma=6.0, cutoff=0.5, min_distance=4)
    mask_1_d = get_mask(norm_1_d, log_sigma=6.0, cutoff=0.4, min_distance=4)

    mask_1 = remove_diagonal_bridges_3d(smart_merge_objects(mask_1_b, mask_1_d))
    
    # === BRANCH 2: Radius 3 (Small/Fine features) ===
    sub_2 = bg_subtraction_3d(roi_img, radius=2)
    
    sub_2 = np.clip(sub_2, 0, None)
    
    norm_2_b = normalize_minmax(sub_2, high_bright=True)
    norm_2_d = normalize_minmax(sub_2, high_bright=False)
    
    mask_2_b = get_mask(norm_2_b, log_sigma=2.0, cutoff=0.5, min_distance=3)
    mask_2_d = get_mask(norm_2_d, log_sigma=2.0, cutoff=0.4, min_distance=3)

    mask_2 = filter_shape(remove_diagonal_bridges_3d(smart_merge_objects(mask_2_b, mask_2_d)))

    # === FINAL MERGE ===
    final_combined = remove_diagonal_bridges_3d(smart_merge_objects(mask_1, mask_2))

    # === DEBUG SAVING ===
    if save_prefix:
        tifffile.imwrite(f"{save_prefix}_debug_mask_1_b.tif", (mask_1_b * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_1_d.tif", (mask_1_d * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2_b.tif", (mask_2_b * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2_d.tif", (mask_2_d * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_1.tif", (mask_1 * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2.tif", (mask_2 * 255).astype(np.uint8))
    
    # Mask OUTSIDE pixels
    final_combined = final_combined & roi_mask    
    
    # Clean up small objects (connectivity=1 is 6-connectivity in 3D)
    final_cleaned = morphology.remove_small_objects(final_combined, min_size=5, connectivity=1)
    
    return final_cleaned

def process_single_file(input_filepath, debug=True):
    """
    Processes a single 3D TIF file.
    Looks for 'rois.zip' in the same directory.
    Saves output 3D masks in the same directory.
    """
    if not input_filepath.exists():
        print(f"Error: File not found: {input_filepath}")
        return

    input_dir = input_filepath.parent
    filename = input_filepath.name
    base_name = input_filepath.stem  

    try:
        # Load Full 3D Image (z, y, x)
        raw_img = tifffile.imread(input_filepath)
        
        rois = roi_tools.load_roi_file(input_filepath)
        if not rois:
            return

        for roi_name, roi_data in rois.items():
            print(f"  > Processing ROI: {roi_name}")
            
            # 3. Create 3D boolean mask for the ROI
            roi_mask = roi_tools.roi_to_mask(roi_data, raw_img.shape)
            
            # 4. Extract 3D Bounding Box
            coords = np.argwhere(roi_mask)
            if coords.size == 0:
                continue
            
            # 3D coordinates (z, y, x)
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0) + 1
            
            # Crop the 3D image
            roi_crop = raw_img[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            roi_mask_crop = roi_mask[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # --- RUN SEGMENTATION ON THE 3D CROP ---
            if np.max(roi_crop) > 0:
                if debug:
                    debug_prefix = str(input_dir / f"{base_name}_{roi_name}")
                else:
                    debug_prefix = None
                
                seg_mask_crop = process_roi_image(roi_crop, roi_mask_crop, save_prefix=debug_prefix)
            else:
                seg_mask_crop = np.zeros_like(roi_crop, dtype=bool)

            # 5. Place crop back into full 3D image size
            final_full_mask = np.zeros_like(raw_img, dtype=bool)
            final_full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = seg_mask_crop
            
            # 6. Save Output (In the SAME folder)
            save_name = f"{base_name}_{roi_name}_mask.tif"
            save_path = input_dir / save_name
            
            save_img = (final_full_mask * 255).astype(np.uint8)
            tifffile.imwrite(save_path, save_img)
            print(f"Saved: {save_name}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    bg_tools.init_imagej()
    
    # Process the single file
    process_single_file(cfg.R_RAW, cfg.DEBUG_MODE)