import os
import shutil

def flatten_directory(target_folder):
    """
    Searches every subfolder in target_folder, moves every file to target_folder,
    and deletes empty subfolders, ignoring '8 bit' and '8bit' folders.
    """
    # specific folder names to ignore (lowercase for case-insensitive matching)
    ignored_names = {'8 bit', '8bit'}

    # Verify the target folder exists
    if not os.path.isdir(target_folder):
        print(f"Error: The directory '{target_folder}' does not exist.")
        return

    print(f"Processing: {target_folder}...")

    # --- STEP 1: Move Files ---
    # topdown=True allows us to modify 'dirs' in-place to exclude folders from being searched
    for root, dirs, files in os.walk(target_folder, topdown=True):
        
        # Modify dirs in-place to skip ignored folders.
        # This prevents the loop from entering '8 bit' or '8bit' directories entirely.
        dirs[:] = [d for d in dirs if d.lower() not in ignored_names]

        # Skip moving files if we are currently in the root target_folder
        if root == target_folder:
            continue

        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(target_folder, file)

            # Check if file already exists in destination to avoid overwriting
            if os.path.exists(destination_path):
                base, extension = os.path.splitext(file)
                counter = 1
                # Rename file (e.g., "image_1.png")
                while os.path.exists(os.path.join(target_folder, f"{base}_{counter}{extension}")):
                    counter += 1
                new_filename = f"{base}_{counter}{extension}"
                destination_path = os.path.join(target_folder, new_filename)
                print(f"Renamed collision: {file} -> {new_filename}")

            try:
                shutil.move(source_path, destination_path)
            except Exception as e:
                print(f"Could not move {source_path}: {e}")

    # --- STEP 2: Delete Empty Subfolders ---
    # topdown=False walks from bottom-up, ensuring we delete child folders before parents
    for root, dirs, files in os.walk(target_folder, topdown=False):
        if root == target_folder:
            continue

        folder_name = os.path.basename(root)
        
        # Do not delete if it is one of the ignored folders
        if folder_name.lower() in ignored_names:
            continue

        # Attempt to remove the directory. os.rmdir only removes EMPTY directories.
        # If the directory still contains files (or the ignored '8 bit' folder), 
        # this will fail safely and silently, preserving the structure.
        try:
            os.rmdir(root)
            # print(f"Deleted empty folder: {root}") # Uncomment to see deleted folders
        except OSError:
            # Directory was not empty
            pass

    print("Operation complete.")

# --- Usage ---
# Replace the path below with your actual folder path
folder_path = r"F:\20250517 PFKL-mCherry_PyronicSF_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 2 - 180 mM" 
flatten_directory(folder_path)