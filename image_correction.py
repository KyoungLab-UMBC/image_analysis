import os
import numpy as np
import tifffile
from scipy.ndimage import shift

# 1. The folder where your image is located
#    (Use forward slashes '/' or double backslashes '\\')
FOLDER_PATH = r"/media/kyoung/Elements/20260101 PFKL-mCherry_EGFP-GAPDH - High Salt Conc - 24.1 degree - LLSM/Deconv/Cell2 - 180mM/3_NoGlucose60min"

# 2. The image file name
FILE_NAME = "Cell2_ex560em593_250mW_200ms_NoGlucose60min_d62.tif"

# 3. The shift vector [X, Y, Z]
SHIFT_VECTOR = [1.77299999999997, 0.199999999999989, 5.601]

# =========================================================

def main():
    # Construct full file paths
    full_input_path = os.path.join(FOLDER_PATH, FILE_NAME)
    
    # Construct output filename: name + _corrected + .tif
    name_part, ext_part = os.path.splitext(FILE_NAME)
    output_filename = f"{name_part}_corrected{ext_part}"
    full_output_path = os.path.join(FOLDER_PATH, output_filename)

    # Check if file exists
    if not os.path.exists(full_input_path):
        print(f"Error: File not found at:\n{full_input_path}")
        print("Please check the FOLDER_PATH and FILE_NAME in the Configuration Section.")
        return

    print(f"Reading file: {FILE_NAME}...")
    try:
        # Read the TIFF stack
        image_stack = tifffile.imread(full_input_path)
    except Exception as e:
        print(f"Failed to open file: {e}")
        return

    # Convert to float64 for precision during shift
    image_float = image_stack.astype(np.float64)

    # Prepare Shift Vector
    # User input: [X, Y, Z]
    # Python Image Stack shape: (Z, Y, X)
    # We must reorder inputs [0, 1, 2] -> [2, 1, 0]
    shift_z = SHIFT_VECTOR[2]
    shift_y = SHIFT_VECTOR[1]
    shift_x = SHIFT_VECTOR[0]
    
    scipy_shift_vector = [shift_z, shift_y, shift_x]

    print(f"Applying shift (Z, Y, X): {scipy_shift_vector}...")

    # Apply Shift
    # order=1 is linear (standard), order=3 is cubic
    corrected_image = shift(image_float, shift=scipy_shift_vector, order=1, mode='constant', cval=0)

    # Clip values to valid range and convert back to uint16
    corrected_image = np.clip(corrected_image, 0, 65535).astype(np.uint16)

    # Save output
    print(f"Saving to: {output_filename}...")
    tifffile.imwrite(
        full_output_path, 
        corrected_image, 
        photometric='minisblack',
        compression=None
    )
    
    print("Processing Complete.")

if __name__ == "__main__":
    main()