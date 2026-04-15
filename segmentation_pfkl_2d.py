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

def dot_2d(struct_img, log_sigma, cutoff=-1):
    """Apply 2D spot filter (LoG)"""
    if struct_img.ndim != 2:
        print(f"Warning: dot_2d received {struct_img.ndim}D image")
    
    # LoG response
    response = -1 * (log_sigma**2) * ndi.gaussian_laplace(struct_img, log_sigma)
    
    if cutoff < 0:
        return response
    else:
        return response > cutoff


def filter_shape(mask, max_ar=2.6, min_area=5, min_solidity=0.85):
    """
    Filters objects by:
    1. Min Area (removes tiny noise < 5 pixels)
    2. Max Area (removes large blobs)(deleted, but could be added back if needed)
    3. Aspect Ratio (removes lines)
    4. Solidity (removes jagged/worm shapes)
    """
    labeled_img = measure.label(mask)
    regions = measure.regionprops(labeled_img)
    new_mask = np.zeros_like(mask, dtype=bool)
    
    for region in regions:
        # 1. Min Area Filter (NEW)
        if region.area < min_area: continue
        
        # 3. Aspect Ratio Filter
        if region.minor_axis_length == 0:
            ar = 9999.0
        else:
            ar = region.major_axis_length / region.minor_axis_length
            
        if ar > max_ar: continue

        # 4. Solidity Filter (Keeps compact dots, removes jagged lines)
        if region.solidity < min_solidity: continue
            
        coords = region.coords
        new_mask[coords[:,0], coords[:,1]] = True
        
    return new_mask

def remove_diagonal_bridges(binary_mask):
    """
    Detects diagonal connections (8-connectivity) and breaks them.
    Uses array slicing (views) for robust, explicit neighbor checking.
    """
    # Create views for the 2x2 sliding window components
    # TL = Top-Left, TR = Top-Right, BL = Bottom-Left, BR = Bottom-Right
    # These views are references, so modifying them modifies the original mask.
    TL = binary_mask[:-1, :-1]
    TR = binary_mask[:-1, 1:]
    BL = binary_mask[1:, :-1]
    BR = binary_mask[1:, 1:]
    
    # --- Pattern A: Top-Left to Bottom-Right connection ---
    # Pattern: 1 0
    #          0 1
    # We detect where TL==1, TR==0, BL==0, BR==1
    mask_a = (TL == True) & (TR == False) & (BL == False) & (BR == True)
    
    # Break the bridge by deleting the Bottom-Right pixel (BR)
    # You could delete TL instead, but we just need to break one.
    BR[mask_a] = False
    
    # --- Pattern B: Top-Right to Bottom-Left connection (YOUR ISSUE) ---
    # Pattern: 0 1
    #          1 0
    # We detect where TL==0, TR==1, BL==1, BR==0
    mask_b = (TL == False) & (TR == True) & (BL == True) & (BR == False)
    
    # Break the bridge by deleting the Bottom-Left pixel (BL)
    # This specifically targets the '1' at the bottom-left of the junction.
    BL[mask_b] = False
    
    return binary_mask

def separate_attached_objects_watershed(binary_mask, min_distance):
    """
    Separates attached objects in a binary mask using Watershed.
    Returns a binary mask with lines separating the objects.
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
        
    # 1. Distance Transform
    distance = ndi.distance_transform_edt(binary_mask)
    
    # 2. Find Peaks (Markers)
    # min_distance controls how close two peaks can be to be considered separate
    coords = feature.peak_local_max(distance, min_distance= min_distance, labels=binary_mask)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    
    markers, _ = ndi.label(mask_peaks)
    
    # 3. Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary_mask, watershed_line=True)
    
    # 4. Convert back to binary (labels > 0)
    # Note: Watershed lines (0) will separate the objects
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
    labeled_union = measure.label(union_mask, connectivity=2)
    labeled_small = measure.label(mask_small, connectivity=2)
    labeled_large = measure.label(mask_large, connectivity=2)
    
    # Initialize the marker array
    markers = np.zeros_like(union_mask, dtype=np.int32)
    current_label = 1
    
    # 2. For each object in the union_mask
    for region in measure.regionprops(labeled_union):
        r_slice = region.slice
        union_obj = region.image
        
        local_mask_large = mask_large[r_slice]
        local_mask_small = mask_small[r_slice]
        
        # Isolate small and large labels specifically within this union object's bounding box
        local_small_labels = labeled_small[r_slice].copy()
        local_small_labels[~union_obj] = 0
        
        local_large_labels = labeled_large[r_slice].copy()
        local_large_labels[~union_obj] = 0
        
        area_union = region.area
        markers_in_this_union = 0
        
        # View of the marker array for this bounding box
        m_slice = markers[r_slice]
        
        # --- For each object in the mask_small inside this object in union_mask ---
        small_ids = np.unique(local_small_labels)
        small_ids = small_ids[small_ids > 0]
        
        for sid in small_ids:
            small_obj = (local_small_labels == sid)
            area_small = np.sum(small_obj)
            area_overlap = np.sum(small_obj & local_mask_large)
            
            if area_small > 0:
                ratio_overlap = area_overlap / area_small
                ratio_union = area_small / area_union
                
                # If overlap with large < 0.2 OR area ratio to union > 0.2, count as marker
                if ratio_union >= 0.1:
                    m_slice[small_obj] = current_label
                    current_label += 1
                    markers_in_this_union += 1
                    
        # --- For each object in the mask_large inside this object in union_mask ---
        large_ids = np.unique(local_large_labels)
        large_ids = large_ids[large_ids > 0]
        
        for lid in large_ids:
            large_obj = (local_large_labels == lid)
            area_large = np.sum(large_obj)
            
            # The area of objects in mask_small overlapped with this object in mask_large
            overlap_mask = large_obj & local_mask_small
            area_overlap = np.sum(overlap_mask)
            
            if area_large > 0:
                ratio_overlap = area_overlap / area_large
                ratio_large_union = area_large / area_union
                # If overlap ratio < 0.3, count as marker
                if ratio_overlap <= 0.15 and ratio_large_union <= 0.6:
                    # Only assign marker to pixels that haven't already been claimed by a mask_small marker
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                # If this object is only one part of the union
                elif ratio_large_union <= 0.25:
                    # Only assign marker to pixels that haven't already been claimed by a mask_small marker
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                # If the total area of mask_small objects inside this large object is < 0.3 of its area, delete the markers previously assigned from mask_small
                elif ratio_overlap <= 0.15 and ratio_large_union > 0.91:
                    # Find the specific marker IDs generated by mask_small that overlap with this large object
                    overlapping_marker_ids = np.unique(m_slice[overlap_mask])                    
                    # Delete those entire markers from the slice
                    for m_id in overlapping_marker_ids:
                        if m_id > 0:  # Ignore 0 (background)
                            m_slice[m_slice == m_id] = 0
                # --- NEW LOGIC ---
                # Fallback: if there is no marker inside from mask_small or mask_large at last, add one
                # We check if this specific large object has any marked pixels (m_slice > 0)
                if not np.any(large_obj & (m_slice > 0)):
                    free_pixels = large_obj & (m_slice == 0)
                    if np.any(free_pixels):
                        m_slice[free_pixels] = current_label
                        current_label += 1
                        markers_in_this_union += 1
                # else ignore it
                
        # Safety Fallback: If no markers were assigned (all objects ignored), 
        # keep the whole union object as a single marker to prevent it from disappearing.
        if markers_in_this_union == 0:
            free_pixels = union_obj & (m_slice == 0)
            if np.any(free_pixels):
                m_slice[free_pixels] = current_label
                current_label += 1
                
    # 3. Do the watershed with the markers above to union_mask
    distance = ndi.distance_transform_edt(union_mask)
    
    # SHRINk MARKERS to single points (peaks) so watershed has room to grow and draw boundaries
    # We find the center point of each existing marker to use as the true watershed seed
    shrunk_markers = np.zeros_like(markers)
    for marker_id in np.unique(markers)[1:]: # Skip 0 (background)
        # Find the pixel in this marker region furthest from the background
        marker_mask = (markers == marker_id)
        local_dist = distance * marker_mask
        
        # Get coordinates of the maximum distance pixel
        max_idx = np.argmax(local_dist)
        coords = np.unravel_index(max_idx, local_dist.shape)
        
        # Place a single point marker
        shrunk_markers[coords] = marker_id
        
    # Run watershed using the SHRUNK markers
    labels = segmentation.watershed(-distance, shrunk_markers, mask=union_mask, watershed_line=True)
    
    return labels > 0

def get_mask(img, log_sigma, cutoff, min_distance):
    """Helper function to apply dot_2d and filter_shape in one step"""
    mask_1 = dot_2d(img, log_sigma, cutoff)
    mask_2 = dot_2d(img, log_sigma/2, cutoff/2)
    mask_max = (img == 1.0)

    merged = mask_1 | mask_2 | mask_max
    mask = remove_diagonal_bridges(separate_attached_objects_watershed(merged, min_distance))
    return mask

def process_roi_image(roi_img, roi_mask, save_prefix=None):
    """
    Applies the specific 2-branch logic to a single cropped ROI image.
    """
    # === BRANCH 1: Radius 20 (Large/Medium features) ===
    # 1. Background Subtract (Radius 20)
    # CHANGED: Get background-subtracted image DIRECTLY (create_background=False)
    # This replaces the manual subtraction step.
    sub_1 = bg_tools.estimate_background_rolling_ball(roi_img, radius=10, create_background=False)
    
    # Ensure float32 for processing and clip any potential negatives
    sub_1 = sub_1.astype(np.float32)
    sub_1 = np.clip(sub_1, 0, None)
    
    # 2. Normalize, b for bright, d for dim
    norm_1_b = normalize_minmax(sub_1, high_bright=True)
    norm_1_d = normalize_minmax(sub_1, high_bright=False)
    # 
    mask_1_b = get_mask(norm_1_b, log_sigma=6.0, cutoff=0.5, min_distance=4)
    mask_1_d = get_mask(norm_1_d, log_sigma=6.0, cutoff=0.4, min_distance=4)

    mask_1 = remove_diagonal_bridges(smart_merge_objects(mask_1_b, mask_1_d))
    # === BRANCH 2: Radius 3 (Small/Fine features) ===
    # 1. Background Subtract (Radius 3)
    # CHANGED: Get background-subtracted image DIRECTLY
    sub_2 = bg_tools.estimate_background_rolling_ball(roi_img, radius=2, create_background=False)
    
    sub_2 = sub_2.astype(np.float32)
    sub_2 = np.clip(sub_2, 0, None)
    
    # 3. Dot Detect
    norm_2_b = normalize_minmax(sub_2, high_bright=True)
    norm_2_d = normalize_minmax(sub_2, high_bright=False)
    # 
    mask_2_b = get_mask(norm_2_b, log_sigma=2.0, cutoff=0.5, min_distance=3)
    mask_2_d = get_mask(norm_2_d, log_sigma=2.0, cutoff=0.4, min_distance=3)

    mask_2 = filter_shape(remove_diagonal_bridges(smart_merge_objects(mask_2_b, mask_2_d)))

    # === FINAL MERGE ===
    final_combined = remove_diagonal_bridges(smart_merge_objects(mask_1, mask_2))

    # === NEW: DEBUG SAVING ===
    if save_prefix:
        tifffile.imwrite(f"{save_prefix}_debug_mask_1_b.tif", (mask_1_b * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_1_d.tif", (mask_1_d * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2_b.tif", (mask_2_b * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2_d.tif", (mask_2_d * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_1.tif", (mask_1 * 255).astype(np.uint8))
        tifffile.imwrite(f"{save_prefix}_debug_mask_2.tif", (mask_2 * 255).astype(np.uint8))
    
    # Mask OUTSIDE pixels
    final_combined = final_combined & roi_mask    
    
    # Clean up small objects (connectivity=1 is now natural)
    final_cleaned = morphology.remove_small_objects(final_combined, min_size=5, connectivity=1)
    
    return final_cleaned

def process_single_file(input_filepath, debug=True):
    """
    Processes a single 16-bit TIF file.
    Looks for 'rois.zip' in the same directory.
    Saves output masks in the same directory.
    """
    # 1. Get path information
    if not input_filepath.exists():
        print(f"Error: File not found: {input_filepath}")
        return

    input_dir = input_filepath.parent
    filename = input_filepath.name
    base_name = input_filepath.stem  # Replaces os.path.splitext(filename)[0]   

    try:
        # Load Full Image
        raw_img = tifffile.imread(input_filepath)
        
        rois = roi_tools.load_roi_file(input_filepath)
        if not rois:
            return

        for roi_name, roi_data in rois.items():
            print(f"  > Processing ROI: {roi_name}")
            
            # 3. Create boolean mask for the ROI
            roi_mask = roi_tools.roi_to_mask(roi_data, raw_img.shape)
            
            # 4. Extract Bounding Box
            coords = np.argwhere(roi_mask)
            if coords.size == 0:
                continue
            
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0) + 1
            
            # Crop the image
            roi_crop = raw_img[r_min:r_max, c_min:c_max].copy()
            roi_mask_crop = roi_mask[r_min:r_max, c_min:c_max]
            
            # --- RUN SEGMENTATION ON THE CROP ---
            if np.max(roi_crop) > 0:
                if debug:
                    # Construct prefix as Path, then convert to str for string concatenation in sub-functions
                    debug_prefix = str(input_dir / f"{base_name}_{roi_name}")
                else:
                    debug_prefix = None
                
                # If debug_prefix is None, process_roi_image will NOT save the intermediate TIFs
                seg_mask_crop = process_roi_image(roi_crop, roi_mask_crop, save_prefix=debug_prefix)
            else:
                seg_mask_crop = np.zeros_like(roi_crop, dtype=bool)

            # 5. Place crop back into full image size
            final_full_mask = np.zeros_like(raw_img, dtype=bool)
            final_full_mask[r_min:r_max, c_min:c_max] = seg_mask_crop
            
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