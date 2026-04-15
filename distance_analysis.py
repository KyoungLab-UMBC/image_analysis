import numpy as np
import pandas as pd
from skimage import io, measure, segmentation
from scipy.spatial import cKDTree
import os

# ==========================================
# 1. SETUP & INPUTS
# ==========================================

# 1. Base Path: The folder containing input files and where output will be saved
base_path = r'F:\20250517 PFKL-mCherry_Queen37C_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM\Cell6\1_AddSalt45min'

# 2. Input File Names (Must be inside the base_path)
mask_a_name = 'Cell0205_1_mask.tif'
mask_b_name = 'Cell0208_1_mask.tif'

# Construct full paths automatically
mask_a_path = os.path.join(base_path, mask_a_name)
mask_b_path = os.path.join(base_path, mask_b_name)

# 3. Output Configuration
output_filename = "Distance Analysis.xlsx"
output_path = os.path.join(base_path, output_filename)

# Name variables for the Excel sheets
name_a = "PFKL"
name_b = "Mitochondria"

# Boolean: True for 2D images, False for 3D stacks
is_2d = True

# Scalar setup (nm/pixel or um/pixel)
# scalar_z is ignored if is_2d is True
scalar_xy = 107
scalar_z = 187

# ==========================================
# CORE FUNCTIONS
# ==========================================

def get_object_data(mask_path, is_2d_mode):
    """
    Loads a mask, identifies objects, and extracts their surface coordinates 
    and centroids.
    """
    print(f"Processing {mask_path}...")
    
    # Load image
    try:
        img = io.imread(mask_path)
    except FileNotFoundError:
        print(f"Error: File {mask_path} not found.")
        return [], []

    # Ensure binary
    img = (img > 0).astype(int)
    
    # Label connected components
    labeled_img = measure.label(img, connectivity=2)
    regions = measure.regionprops(labeled_img)
    
    objects_data = []
    all_surface_points = []
    
    for props in regions:
        # Get Centroid
        centroid = props.centroid
        
        # Get bounding box coordinates
        if is_2d_mode:
            min_row, min_col, max_row, max_col = props.bbox
            
            # Crop image
            obj_crop = props.image
            
            # --- FIX: PAD THE CROP ---
            # Pad with 1 pixel of 0s on all sides so find_boundaries sees the edges
            obj_crop_padded = np.pad(obj_crop, pad_width=1, mode='constant', constant_values=0)
            
            # Find boundaries on padded image
            boundary_padded = segmentation.find_boundaries(obj_crop_padded, mode='inner')
            y_pad, x_pad = np.where(boundary_padded)
            
            # Adjust coordinates back to original image space
            # (Subtract 1 to account for padding, then add bbox offset)
            y_global = (y_pad - 1) + min_row
            x_global = (x_pad - 1) + min_col
            
            # Store points
            obj_points = np.column_stack((y_global, x_global))
            
            # Format Centroid
            centroid_export = (centroid[1], centroid[0], 0) 
            
        else:
            # 3D Logic
            min_d, min_r, min_c = props.bbox[0], props.bbox[1], props.bbox[2]
            
            # Crop image
            obj_crop = props.image
            
            # --- FIX: PAD THE CROP ---
            obj_crop_padded = np.pad(obj_crop, pad_width=1, mode='constant', constant_values=0)
            
            # Find boundaries
            boundary_padded = segmentation.find_boundaries(obj_crop_padded, mode='inner')
            z_pad, y_pad, x_pad = np.where(boundary_padded)
            
            # Adjust coordinates
            z_global = (z_pad - 1) + min_d
            y_global = (y_pad - 1) + min_r
            x_global = (x_pad - 1) + min_c
            
            # Store points
            obj_points = np.column_stack((z_global, y_global, x_global))
            
            # Format Centroid
            centroid_export = (centroid[2], centroid[1], centroid[0])

        # Save object info
        objects_data.append({
            'label': props.label,
            'centroid': centroid_export, 
            'points': obj_points
        })
        
        all_surface_points.append(obj_points)

    # Concatenate all points
    if all_surface_points:
        total_surface_cloud = np.vstack(all_surface_points)
    else:
        total_surface_cloud = np.empty((0, 3 if not is_2d_mode else 2))
        
    return objects_data, total_surface_cloud


def analyze_distances(objects_source, cloud_target, is_2d_mode, s_xy, s_z):
    """
    Calculates the minimum distance from every object in source to the 
    nearest pixel in the target point cloud.
    """
    print("Calculating distances...")
    
    # Handle case where target cloud is empty
    if len(objects_source) == 0 or len(cloud_target) == 0:
        return [float('nan')] * len(objects_source)

    # 1. Apply scaling to the TARGET cloud
    scaled_target = cloud_target.astype(float).copy()
    
    if is_2d_mode:
        scaled_target[:, 0] *= s_xy
        scaled_target[:, 1] *= s_xy
    else:
        scaled_target[:, 0] *= s_z
        scaled_target[:, 1] *= s_xy
        scaled_target[:, 2] *= s_xy

    # 2. Build KDTree for target
    tree = cKDTree(scaled_target)
    
    distances = []
    
    for obj in objects_source:
        points = obj['points'].astype(float).copy()
        
        # --- FIX START: Check if object has points ---
        if points.size == 0:
            # If an object has no boundary points (e.g., artifact), append NaN
            distances.append(float('nan')) 
            continue
        # --- FIX END ---

        # Apply scaling to SOURCE object points
        if is_2d_mode:
            points[:, 0] *= s_xy
            points[:, 1] *= s_xy
        else:
            points[:, 0] *= s_z
            points[:, 1] *= s_xy
            points[:, 2] *= s_xy
            
        # 3. Query the tree
        dists, _ = tree.query(points, k=1)
        
        # Calculate min only if we have distances
        min_dist = np.min(dists)
        distances.append(min_dist)
        
    return distances


def export_to_excel(data_a, data_b, dists_a, dists_b, filename="Results.xlsx"):
    """
    Exports results to Excel with specified formatting.
    """
    print(f"Exporting to {filename}...")
    
    # Define the exact headers you want
    headers = ['Object Number', 'Centroid X', 'Centroid Y', 'Centroid Z', 'Nearest Distance']

    # Prepare DataFrame A
    rows_a = []
    for i, obj in enumerate(data_a):
        rows_a.append({
            headers[0]: obj['label'],
            headers[1]: obj['centroid'][0],
            headers[2]: obj['centroid'][1],
            headers[3]: obj['centroid'][2],
            headers[4]: dists_a[i]
        })
    df_a = pd.DataFrame(rows_a)
    
    # Prepare DataFrame B
    rows_b = []
    for i, obj in enumerate(data_b):
        rows_b.append({
            headers[0]: obj['label'],
            headers[1]: obj['centroid'][0],
            headers[2]: obj['centroid'][1],
            headers[3]: obj['centroid'][2],
            headers[4]: dists_b[i]
        })
    df_b = pd.DataFrame(rows_b)

    # Write to Excel
    # We remove header=False and the manual worksheet loop. 
    # Letting pandas handle the headers prevents the AttributeError.
    with pd.ExcelWriter(filename) as writer:
        df_a.to_excel(writer, sheet_name=name_a, index=False)
        df_b.to_excel(writer, sheet_name=name_b, index=False)

    print("Done.")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # 1. Load and Process Masks
    # (Uses the constructed paths from Setup)
    objects_a, cloud_a = get_object_data(mask_a_path, is_2d)
    objects_b, cloud_b = get_object_data(mask_b_path, is_2d)
    
    if objects_a and objects_b:
        # 2. Calculate Distances
        dists_a_to_b = analyze_distances(objects_a, cloud_b, is_2d, scalar_xy, scalar_z)
        
        dists_b_to_a = analyze_distances(objects_b, cloud_a, is_2d, scalar_xy, scalar_z)
        
        # 3. Export
        # We pass the full output_path defined in Setup
        export_to_excel(objects_a, objects_b, dists_a_to_b, dists_b_to_a, filename=output_path)
    else:
        print("Could not process one or both files. Check file paths.")