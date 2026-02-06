import os
import glob
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology, draw, segmentation, feature
from read_roi import read_roi_zip
import background_correction as bg_tools



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

def normalize_minmax(img):
    """Normalize array to 0-1 range"""
    img = img.astype(np.float32)
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v - min_v == 0:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)

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

def separate_attached_objects_watershed(binary_mask):
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
    coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_mask)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    
    markers, _ = ndi.label(mask_peaks)
    
    # 3. Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary_mask)
    
    # 4. Convert back to binary (labels > 0)
    # Note: Watershed lines (0) will separate the objects
    return labels > 0

def roi_to_mask(roi, image_shape):
    mask = np.zeros(image_shape, dtype=bool)
    roi_type = roi['type']
    
    try:
        if roi_type == 'rectangle':
            r_start = roi['top']
            r_end = roi['top'] + roi['height']
            c_start = roi['left']
            c_end = roi['left'] + roi['width']
            mask[r_start:r_end, c_start:c_end] = True
        elif roi_type in ['polygon', 'freehand', 'traced']:
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
        print(f"Warning: ROI conversion failed: {e}")
        
    return mask

def process_roi_image(roi_img, roi_mask):
    """
    Applies the specific 2-branch logic to a single cropped ROI image.
    """
    # === BRANCH 1: Radius 20 (Large/Medium features) ===
    # 1. Background Subtract (Radius 20)
    bg_1 = bg_tools.estimate_background_rolling_ball(roi_img, radius=20)
    # Subtract safely
    sub_1 = roi_img.astype(np.float32) - bg_1.astype(np.float32)
    sub_1 = np.clip(sub_1, 0, None) # Clip negatives
    
    # 2. Normalize
    norm_1 = normalize_minmax(sub_1)
    
    # 3. Dot Detect
    mask_1 = dot_2d(norm_1, log_sigma=4.0, cutoff=0.1)
    mask_2 = dot_2d(norm_1, log_sigma=2.0, cutoff=0.05)
    
    # 4. Merge & Watershed
    merged_1 = mask_1 | mask_2
    mask_a = separate_attached_objects_watershed(merged_1)

    # === BRANCH 2: Radius 3 (Small/Fine features) ===
    # 1. Background Subtract (Radius 3)
    bg_2 = bg_tools.estimate_background_rolling_ball(roi_img, radius=3)
    sub_2 = roi_img.astype(np.float32) - bg_2.astype(np.float32)
    sub_2 = np.clip(sub_2, 0, None)
    
    # 2. Normalize
    norm_2 = normalize_minmax(sub_2)
    
    # 3. Dot Detect
    mask_3 = dot_2d(norm_2, log_sigma=2.0, cutoff=0.1)
    
    raw_mask_4 = dot_2d(norm_2, log_sigma=1.0, cutoff=0.05)
    mask_4 = filter_shape(raw_mask_4, max_area=50, max_ar=2.5)
    
    # 4. Merge & Watershed
    merged_2 = mask_3 | mask_4
    mask_b = separate_attached_objects_watershed(merged_2)

    # === FINAL MERGE ===
    final_combined = mask_a | mask_b
    # This prevents small fragments cut by the ROI border from being deleted if they were part of a larger object
    final_combined = final_combined & roi_mask
    # Discard objects < 5 pixels
    final_cleaned = morphology.remove_small_objects(final_combined, min_size=5)
    
    return final_cleaned

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Check all TIF files
    all_files = glob.glob(os.path.join(input_folder, "*.tif"))
    raw_files = [f for f in all_files if "mask" not in f.lower()]
    
    print(f"Found {len(raw_files)} raw images in {input_folder}")

    for filepath in raw_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        # 2. Find suffix "_rois.zip"
        roi_zip_path = os.path.join(input_folder, f"{base_name}_rois.zip")
        
        if not os.path.exists(roi_zip_path):
            print(f"Skipping {filename}: ROI file not found ({base_name}_rois.zip)")
            continue

        print(f"Processing {filename} with ROIs...")

        try:
            # Load Full Image
            raw_img = tifffile.imread(filepath)
            
            # Read ROIs
            rois = read_roi_zip(roi_zip_path)
            if not rois:
                print("  > Zip file contained no ROIs.")
                continue

            for roi_name, roi_data in rois.items():
                print(f"  > Processing ROI: {roi_name}")
                
                # 3. Create boolean mask for the ROI
                roi_mask = roi_to_mask(roi_data, raw_img.shape)
                
                # 4. Extract Bounding Box (To create a smaller "duplicate" image)
                # finding the bounding box allows us to process a smaller array
                coords = np.argwhere(roi_mask)
                if coords.size == 0:
                    continue
                
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0) + 1 # +1 for slice inclusive
                
                # Crop the image (This acts as the "Duplicate")
                roi_crop = raw_img[r_min:r_max, c_min:c_max].copy()
                roi_mask_crop = roi_mask[r_min:r_max, c_min:c_max]
                
                # --- RUN SEGMENTATION ON THE CROP ---
                # We only process if there is actual data
                if np.max(roi_crop) > 0:
                    # CHANGED: Pass the mask into the function
                    seg_mask_crop = process_roi_image(roi_crop, roi_mask_crop)
                else:
                    seg_mask_crop = np.zeros_like(roi_crop, dtype=bool)

                # 5. Place crop back into full image size for saving
                final_full_mask = np.zeros_like(raw_img, dtype=bool)
                final_full_mask[r_min:r_max, c_min:c_max] = seg_mask_crop
                
                # 6. Save
                save_name = f"{base_name}_{roi_name}_mask.tif"
                save_path = os.path.join(output_folder, save_name)
                
                save_img = (final_full_mask * 255).astype(np.uint8)
                tifffile.imwrite(save_path, save_img)
                print(f"    Saved: {save_name}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_dir = r"/media/kyoung/Elements1/20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField/Plate 2 - 180 mM/Cell6/1_AddSalt45min/Red input"
    output_dir = r"/media/kyoung/Elements1/20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField/Plate 2 - 180 mM/Cell6/1_AddSalt45min/Output"
    bg_tools.init_imagej()
    process_folder(input_dir, output_dir)