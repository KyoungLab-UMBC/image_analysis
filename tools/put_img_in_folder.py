import os
import shutil

# ======================
# USER SETTINGS
# ======================
src_dir = r"F:\20250517 PFKL-mCherry_PyronicSF_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM"      # source folder with all images
dst_dir = src_dir      # destination folder

n = 10   # cells
t = 2   # time points
a = 4    # images per time point per cell

timepoint_folders = [
    "0",
    "1_AddSalt45min",
]

# Define the new subfolders here
sub_folders = ["Red input", "Green input", "Output"]

image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ======================
# LOAD AND SORT FILES
# ======================
files = sorted(
    f for f in os.listdir(src_dir)
    if f.lower().endswith(image_exts)
)

expected = n * t * a
if len(files) != expected:
    raise ValueError(
        f"Expected {expected} images, found {len(files)}"
    )

# ======================
# CREATE FOLDER STRUCTURE
# ======================
for cell in range(1, n + 1):
    for tp_name in timepoint_folders:
        # Iterate through the required subfolders and create them
        for sub in sub_folders:
            os.makedirs(
                os.path.join(dst_dir, f"Cell{cell}", tp_name, sub),
                exist_ok=True
            )

# ======================
# ASSIGN FILES BY POSITION
# ======================
idx = 0
for tp in range(t):           # time points outer loop
    for cell in range(1, n + 1):   # cells inner loop
        for _ in range(a):    # images per time point
            filename = files[idx]

            # Note: Files are currently moved to the timepoint folder, not the subfolders.
            dst_path = os.path.join(
                dst_dir,
                f"cell{cell}",
                timepoint_folders[tp],
                filename
            )

            shutil.move(
                os.path.join(src_dir, filename),
                dst_path
            )

            idx += 1

print("✅ Images organized strictly by ascending sequence with new subfolders created.")