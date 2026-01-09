import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import measure
from pptx import Presentation
from pptx.util import Inches
import io
import warnings
import sys
import os
from scipy import signal
from numba import jit
import concurrent.futures

# Set matplotlib backend to Agg for thread safety (no GUI)
plt.switch_backend('Agg')

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- ACCELERATED WORKER FUNCTIONS (NUMBA) ---

@jit(nopython=True)
def calculate_spherical_profile(crop_r, crop_g, crop_mask, obj_id):
    """
    Numba-accelerated function to calculate radial profile in 3D.
    Iterates over the 21x21x21 cube and bins pixels by spherical radius.
    """
    d, h, w = crop_r.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    
    # Radius 0 to 10 (11 bins)
    max_r = 11
    sums_r = np.zeros(max_r)
    sums_g = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    for z in range(d):
        for y in range(h):
            for x in range(w):
                # Mask Logic: Include if background (0) or current object (obj_id)
                # Exclude if it belongs to a neighbor object
                if crop_mask[z, y, x] != 0 and crop_mask[z, y, x] != obj_id:
                    continue
                
                # 3D Distance
                dist = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
                
                # Floor to get bin index (0.5 -> 0, 1.9 -> 1)
                r_idx = int(dist)
                
                if r_idx < max_r:
                    sums_r[r_idx] += crop_r[z, y, x]
                    sums_g[r_idx] += crop_g[z, y, x]
                    counts[r_idx] += 1
                    
    means_r = np.zeros(max_r)
    means_g = np.zeros(max_r)
    
    for i in range(max_r):
        if counts[i] > 0:
            means_r[i] = sums_r[i] / counts[i]
            means_g[i] = sums_g[i] / counts[i]
            
    return means_r, means_g

# --- HELPER FUNCTIONS ---

def analyze_profile_trend(norm_r, norm_g):
    """
    Classifies relationship. Adjusted for shorter profile (len 11).
    """
    # Pearson Correlation
    if len(norm_r) > 2:
        r_value = np.corrcoef(norm_r, norm_g)[0, 1]
    else:
        r_value = 0 
        
    if np.isnan(r_value): r_value = 0

    prominence = np.ptp(norm_g) / 5 if np.ptp(norm_g) > 0 else 0
    peaks, _ = signal.find_peaks(norm_g, prominence=prominence)

    # Classification logic (adjusted indices for smaller range)
    if r_value >= 0.75 or (len(norm_g) > 3 and norm_g[0] > norm_g[1] and norm_g[1] > norm_g[2]):
        return "Colocalized"
    elif r_value <= -0.75:
        return "Anticolocalized"
    elif np.any((peaks >= 1) & (peaks <= 5)):
        return "Around"
    else:
        return "No trend"

def create_plot_image(red_data, green_data, x_axis, obj_id):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    ax1.set_xlabel('Radius (pixel)')
    ax1.set_xlim(0, 10)
    ax1.set_xticks(range(0, 11, 1))
    
    # Left Y: Red
    ax1.set_ylabel('Intensity (Red)', color='red')
    ax1.plot(x_axis, red_data, color='red', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='red')

    # Right Y: Green
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Intensity (Green)', color='green')
    ax2.plot(x_axis, green_data, color='green', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor='green')

    if obj_id != "AVERAGE":
        plt.title(f"Object: {obj_id}")
    
    fig.tight_layout()
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=100)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

def create_composite_z_proj_thumbnail(crop_r, crop_g):
    """
    Generates a Z-Projection (Max Intensity) thumbnail.
    """
    # Z-Projection: Max Intensity
    proj_r = np.max(crop_r, axis=0)
    proj_g = np.max(crop_g, axis=0)
    
    h, w = proj_r.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    def scale_to_uint8(arr):
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 0.00001:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - vmin) / (vmax - vmin) * 255
        return scaled.astype(np.uint8)

    rgb[..., 0] = scale_to_uint8(proj_r) 
    rgb[..., 1] = scale_to_uint8(proj_g) 
    
    img_stream = io.BytesIO()
    plt.imsave(img_stream, rgb, format='png')
    img_stream.seek(0)
    return img_stream

def process_single_object(data_package):
    """
    Worker function for parallel processing.
    """
    (obj_id, cz, cy, cx, area, crop_r, crop_g, crop_mask) = data_package
    
    # --- 1. Calculate Profile (Using Numba) ---
    # Note: Rolling ball is removed. Using raw intensity.
    final_r, final_g = calculate_spherical_profile(crop_r, crop_g, crop_mask, int(obj_id))
    
    # --- 2. Classification ---
    # Arbitrary size threshold, adjust if needed
    size_class = "small" if area <= 1000 else "large" 
    trend_class = analyze_profile_trend(final_r, final_g)
    
    # --- 3. Plotting ---
    x_vals = np.arange(0, 11)
    plot_bytes = create_plot_image(final_r, final_g, x_vals, obj_id).getvalue()
    
    # --- 4. Create Z-Projected Thumbnail ---
    thumb_bytes = create_composite_z_proj_thumbnail(crop_r, crop_g).getvalue()
    
    return {
        "id": obj_id,
        "cz": cz, "cy": cy, "cx": cx,
        "area": area,
        "norm_r": final_r,
        "norm_g": final_g,
        "size_class": size_class,
        "trend_class": trend_class,
        "plot_bytes": plot_bytes,
        "thumb_bytes": thumb_bytes
    }

# --- MAIN CONTROLLER ---

def main(red_path, green_path, mask_path, output_pptx):
    print(f"Loading 3D images...\nRed: {os.path.basename(red_path)}")
    
    try:
        # Load 3D stacks (Z, Y, X)
        img_r = tifffile.imread(red_path)
        img_g = tifffile.imread(green_path)
        mask = tifffile.imread(mask_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Handle extra dimensions if present (e.g., channels)
    if mask.ndim > 3: mask = mask.squeeze()

    # Ensure binary mask is labeled
    unique_vals = np.unique(mask)
    if len(unique_vals) <= 2 or mask.max() == 255:
        print("Labeling binary mask in 3D...")
        labeled_mask = measure.label(mask > 0)
    else:
        labeled_mask = mask

    props = measure.regionprops(labeled_mask)
    print(f"Found {len(props)} objects. Preparing tasks...")

    tasks = []
    d_img, h_img, w_img = img_r.shape
    
    # Define Crop Size (Radius 10 -> Diameter 21)
    crop_rad = 10 
    
    for prop in props:
        cz, cy, cx = map(int, prop.centroid)
        
        # Calculate 3D bounds
        z1, z2 = max(0, cz - crop_rad), min(d_img, cz + crop_rad + 1)
        y1, y2 = max(0, cy - crop_rad), min(h_img, cy + crop_rad + 1)
        x1, x2 = max(0, cx - crop_rad), min(w_img, cx + crop_rad + 1)
        
        # Pad crops if they are near the edge to ensure consistent shape for Numba (optional but safer)
        # For now, we pass whatever slice we get; the Numba function reads dimensions dynamically.
        
        crop_r = img_r[z1:z2, y1:y2, x1:x2]
        crop_g = img_g[z1:z2, y1:y2, x1:x2]
        crop_m = labeled_mask[z1:z2, y1:y2, x1:x2]
        
        tasks.append((
            prop.label, cz, cy, cx, prop.area,
            crop_r, crop_g, crop_m
        ))

    # --- PARALLEL PROCESSING ---
    results = []
    total = len(tasks)
    print(f"Processing {total} objects using Multithreading...")

    # Use ProcessPoolExecutor to bypass GIL and use all cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_object, task) for task in tasks]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                results.append(future.result())
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{total}")
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    print("Processing complete. Generating PowerPoint Report...")

    # --- PPT GENERATION ---
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    groups = {}

    # Sort results by ID for consistent output
    results.sort(key=lambda x: x['id'])

    for res in results:
        k = (res['trend_class'], res['size_class'])
        if k not in groups: groups[k] = {'r': [], 'g': []}
        groups[k]['r'].append(res['norm_r'])
        groups[k]['g'].append(res['norm_g'])

        slide = prs.slides.add_slide(blank_layout)
        
        # Graph
        graph_stream = io.BytesIO(res['plot_bytes'])
        slide.shapes.add_picture(graph_stream, left=Inches(0.5), top=Inches(1), width=Inches(7.5), height=Inches(5))
        
        # Thumbnail (Z-Projected)
        thumb_stream = io.BytesIO(res['thumb_bytes'])
        slide.shapes.add_picture(thumb_stream, left=Inches(8), top=Inches(0.5), width=Inches(3), height=Inches(3))
        
        # Text Data
        tx = slide.shapes.add_textbox(left=Inches(8), top=Inches(4), width=Inches(5), height=Inches(3))
        tf = tx.text_frame
        tf.word_wrap = True
        
        lines = [
            f"Object ID: {res['id']}",
            f"Coords(z,y,x): ({res['cz']}, {res['cy']}, {res['cx']})",
            f"Volume(px): {res['area']}",
            f"Trend: {res['trend_class']}"
        ]
        
        for line in lines:
            p = tf.add_paragraph()
            p.text = line
            p.font.size = Inches(0.25)
            if "Trend" in line and res['trend_class'] != "No trend":
                p.font.bold = True

    # Summary Slides
    trend_types = ["Colocalized", "Around", "Anticolocalized", "No trend"]
    # Adjust size types based on your classification logic above
    size_types = ["small", "large"] 
    
    for trend in trend_types:
        for size in size_types:
            key = (trend, size)
            if key in groups and len(groups[key]['r']) > 0:
                slide = prs.slides.add_slide(blank_layout)
                
                avg_r = np.mean(groups[key]['r'], axis=0)
                avg_g = np.mean(groups[key]['g'], axis=0)
                
                plot_stream = create_plot_image(avg_r, avg_g, np.arange(11), "AVERAGE")
                slide.shapes.add_picture(plot_stream, Inches(0.5), Inches(1), height=Inches(5))
                
                tx = slide.shapes.add_textbox(Inches(8), Inches(2.5), Inches(5), Inches(2))
                tx.text_frame.text = f"Group Average\n{trend}\n{size}\n(n={len(groups[key]['r'])})"

    output_path = output_pptx
    while True:
        try:
            prs.save(output_path)
            print(f"Saved: {output_path}")
            break
        except PermissionError:
            input(f"File {output_path} is OPEN. Close it and press ENTER...")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to point to your 3D TIFF files
    # Assuming input is now a 3D stack (Z, Y, X)
    
    base_path = r"F:\20251217 PFKL-mCherry_EGFP-GAPDH - High Salt Conc - 37 degree - WideField\Plate 1 - 180 mM\cell4\3_NoGlucose60min"
    
    # Example Filenames (Ensure these exist as 3D Tiffs)
    # If your files are split (one file per z-slice), you need to stack them first or load as a sequence.
    # This code assumes Multipage TIFF (ImageJ default for 3D stacks)
    
    if os.path.exists(base_path):
        main(
            os.path.join(base_path, "Cell0278_SubBG1k.tif"), # Red 3D Stack
            os.path.join(base_path, "Cell0279_SubBG1k.tif"), # Green 3D Stack
            os.path.join(base_path, "C2-Composite.tif"),      # Mask 3D Stack
            os.path.join(base_path, "Cell0278_3D_Report.pptx")
        )
    else:
        print("Please configure the 'base_path' in the script.")