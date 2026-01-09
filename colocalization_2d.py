import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import measure, morphology
from pptx import Presentation
from pptx.util import Inches
import io
import warnings
import sys
import os
from scipy import signal
from scyjava import jimport

# ---- SETUP JAVA ENV ----
os.environ["JAVA_HOME"] = r"D:\Anaconda\envs\practice\Library"

# --- IMAGEJ IMPORT ---
import imagej

# ---- Initialize ImageJ ----
print("Initializing ImageJ...")
try:
    # You can keep using '2.14.0' or switch back to r'D:\Fiji.app' if you prefer local
    ij = imagej.init('2.14.0', mode='headless')
    print("Success! ImageJ version:", ij.getVersion())
except Exception as e:
    print(f"Initialization Failed: {e}")
    sys.exit()
IJ = jimport('ij.IJ')

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- WORKER FUNCTIONS ---

def estimate_background_rolling_ball(img, radius=10):
    """
    Estimates background using ImageJ.
    """
    try:
        # 1. Convert NumPy -> ImagePlus
        imp_source = ij.py.to_imageplus(img)
        
        # 2. DUPLICATE: Critical! ImageJ modifies in-place. 
        # We must work on a copy to preserve the original 'img'.
        imp = imp_source.duplicate()
        ip = imp.getProcessor()
        
        # 3. Instantiate BackgroundSubtracter directly
        BackgroundSubtracter = jimport('ij.plugin.filter.BackgroundSubtracter')
        bs = BackgroundSubtracter()
        
        # 4. Execute Rolling Ball
        # (ip, radius, createBackground, lightBackground, useParaboloid, doPresmooth, correctCorners)
        # createBackground=True: 'ip' becomes the background image.
        bs.rollingBallBackground(ip, radius, True, False, False, True, True)
        
        # 5. Convert back to NumPy
        # Convert the ImagePlus (imp), not the processor (ip)
        res_xr = ij.py.from_java(imp)
        
        # Force strict NumPy array (removes xarray wrapper that causes indexing errors)
        res_np = np.array(res_xr)
        
        # Handle dimensions (ImageJ sometimes adds channel dims)
        if res_np.ndim > 2:
            res_np = res_np.squeeze()
            
        return res_np.astype(img.dtype)

    except Exception as e:
        print(f"ImageJ Error: {e}")
        # Fallback
        selem = morphology.disk(radius)
        return morphology.opening(img, selem)


def analyze_profile_trend(norm_r, norm_g):
    """
    Classifies relationship based on Pearson Correlation (r) and Peaks.
    """
    limit = min(len(norm_r), 8)
    
    if limit > 2:
        r_value = np.corrcoef(norm_r[:limit], norm_g[:limit])[0, 1]
    else:
        r_value = 0 
        
    if np.isnan(r_value):
        r_value = 0

    prominence = np.ptp(norm_g) / 5
    peaks, _ = signal.find_peaks(norm_g, prominence=prominence)

    if r_value >= 0.75 or (norm_g[0] > norm_g[1] and norm_g[1] > norm_g[2] and norm_g[2] > norm_g[3]) or norm_g[0] == max(norm_g[:7]):
        return "Colocalized"
    elif r_value <= -0.75 or norm_g[0] == min(norm_g[:10]) and np.median(norm_g) - norm_g[0] >= 3 * prominence:
        return "Anticolocalized"
    elif np.any((peaks >= 1) & (peaks <= 5)) or max(norm_g) - max(norm_g[1:3]) <= prominence:
        return "Around"
    else:
        return "No trend"

def create_plot_image(red_data, green_data, x_axis, obj_id):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    ax1.set_xlabel('Radius(pixel)')
    ax1.set_xlim(0, 20)
    ax1.set_xticks(range(0, 21, 5))
    
    # Left Y: Red
    ax1.set_ylabel('Norm. PFKL (Red)', color='red')
    ax1.plot(x_axis, red_data, color='red', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='red')

    # Right Y: Green
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Norm. GAPDH (Green)', color='green')
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

def process_single_object(data_package):
    """
    Worker function
    """
    (obj_id, cy, cx, area, crop_r, crop_g, crop_mask) = data_package
    
    h, w = crop_r.shape
    center_y, center_x = h // 2, w // 2
    
    # --- 1. Background Estimation (Using ImageJ) ---
    bg_r = estimate_background_rolling_ball(crop_r, radius=10)
    # Prevent divide by zero
    bg_r[bg_r == 0] = 1
    norm_img_r = crop_r.astype(float) / bg_r
    
    bg_g = estimate_background_rolling_ball(crop_g, radius=10)
    bg_g[bg_g == 0] = 1
    norm_img_g = crop_g.astype(float) / bg_g
    
    # --- 2. Radial Profile Calculation ---
    y, x = np.indices((h, w))
    radii_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    profile_r = []
    profile_g = []
    
    for r in range(21):
        ring_mask = (radii_map >= r) & (radii_map < r + 1)
        valid_mask = (crop_mask == 0) | (crop_mask == obj_id)
        final_mask = ring_mask & valid_mask
        
        vals_r = norm_img_r[final_mask]
        vals_g = norm_img_g[final_mask]
        
        profile_r.append(np.mean(vals_r) if len(vals_r) > 0 else 0)
        profile_g.append(np.mean(vals_g) if len(vals_g) > 0 else 0)

    final_r = np.array(profile_r)
    final_g = np.array(profile_g)
    
    # --- 3. Classification ---
    size_class = "small glucosome" if area <= 14 else "large glucosome"
    trend_class = analyze_profile_trend(final_r, final_g)
    
    # --- 4. Plotting ---
    x_vals = np.arange(0, 21)
    plot_bytes = create_plot_image(final_r, final_g, x_vals, obj_id).getvalue()
    
    # --- 5. Create Thumbnail ---
    thumb_bytes = create_composite_thumbnail(norm_img_r, norm_img_g).getvalue()
    
    return {
        "id": obj_id,
        "cy": cy, "cx": cx,
        "area": area,
        "norm_r": final_r,
        "norm_g": final_g,
        "size_class": size_class,
        "trend_class": trend_class,
        "plot_bytes": plot_bytes,
        "thumb_bytes": thumb_bytes
    }

def create_composite_thumbnail(norm_r, norm_g):
    h, w = norm_r.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    def scale_to_uint8(arr):
        vmin = arr.min()
        vmax = arr.max()
        if vmax - vmin < 0.00001:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - vmin) / (vmax - vmin) * 255
        return scaled.astype(np.uint8)

    rgb[..., 0] = scale_to_uint8(norm_r) 
    rgb[..., 1] = scale_to_uint8(norm_g) 
    
    img_stream = io.BytesIO()
    plt.imsave(img_stream, rgb, format='png')
    img_stream.seek(0)
    return img_stream

# --- MAIN CONTROLLER ---

def main(red_path, green_path, mask_path, output_pptx):
    print("Loading images...")
    try:
        img_r = tifffile.imread(red_path)
        img_g = tifffile.imread(green_path)
        mask = tifffile.imread(mask_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    if len(mask.shape) > 2: mask = mask[..., 0]

    unique_vals = np.unique(mask)
    if len(unique_vals) <= 2 or mask.max() == 255:
        print("Labeling binary mask...")
        labeled_mask = measure.label(mask > 0)
    else:
        labeled_mask = mask

    props = measure.regionprops(labeled_mask)
    print(f"Found {len(props)} objects. Starting analysis...")

    tasks = []
    h_img, w_img = img_r.shape
    
    for prop in props:
        cy, cx = map(int, prop.centroid)
        r1, r2 = max(0, cy - 30), min(h_img, cy + 30 + 1)
        c1, c2 = max(0, cx - 30), min(w_img, cx + 30 + 1)
        
        tasks.append((
            prop.label, cy, cx, prop.area,
            img_r[r1:r2, c1:c2],
            img_g[r1:r2, c1:c2],
            labeled_mask[r1:r2, c1:c2]
        ))

    # --- PROCESSING (SINGLE THREADED) ---
    print("Processing objects sequentially (Safe Mode)...")
    
    results = []
    total = len(tasks)
    
    for i, task in enumerate(tasks):
        # Run processing directly
        res = process_single_object(task)
        results.append(res)
        
        if (i + 1) % 50 == 0:
            print(f"Analyzed {i + 1}/{total} objects...")

    print("Processing complete. Generating Report...")

    # --- PPT GENERATION ---
    print("Generating Report...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    groups = {}

    for res in results:
        k = (res['trend_class'], res['size_class'])
        if k not in groups: groups[k] = {'r': [], 'g': []}
        groups[k]['r'].append(res['norm_r'])
        groups[k]['g'].append(res['norm_g'])

        slide = prs.slides.add_slide(blank_layout)
        
        graph_stream = io.BytesIO(res['plot_bytes'])
        slide.shapes.add_picture(graph_stream, left=Inches(0.5), top=Inches(1), width=Inches(7.5), height=Inches(5))
        
        thumb_stream = io.BytesIO(res['thumb_bytes'])
        slide.shapes.add_picture(thumb_stream, left=Inches(8), top=Inches(0.5), width=Inches(3), height=Inches(3))
        
        tx = slide.shapes.add_textbox(left=Inches(8), top=Inches(4), width=Inches(5), height=Inches(3))
        tf = tx.text_frame
        tf.word_wrap = True
        
        lines = [
            f"Object ID: {res['id']}",
            f"Coords(x,y): ({res['cx']}, {res['cy']})",
            f"Size Class: {res['size_class']}",
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
    size_types = ["small glucosome", "large glucosome"]
    
    for trend in trend_types:
        for size in size_types:
            key = (trend, size)
            if key in groups and len(groups[key]['r']) > 0:
                slide = prs.slides.add_slide(blank_layout)
                
                avg_r = np.mean(groups[key]['r'], axis=0)
                avg_g = np.mean(groups[key]['g'], axis=0)
                
                plot_stream = create_plot_image(avg_r, avg_g, np.arange(21), "AVERAGE")
                slide.shapes.add_picture(plot_stream, Inches(0.5), Inches(1), height=Inches(5))
                
                tx = slide.shapes.add_textbox(Inches(8), Inches(2.5), Inches(5), Inches(2))
                tx.text_frame.text = f"Group Average\n{trend}\n{size}\n(n={len(groups[key]['r'])})"

    while True:
        try:
            prs.save(output_pptx)
            print(f"Saved: {output_pptx}")
            break
        except PermissionError:
            input(f"File {output_pptx} is OPEN. Close it and press ENTER...")

if __name__ == "__main__":
    # Point this to your specific files
    user_path = r"F:\20251217 PFKL-mCherry_EGFP-GAPDH - High Salt Conc - 37 degree - WideField\Plate 1 - 180 mM\cell4\3_NoGlucose60min"
    
    # Safety check if path exists
    if not os.path.exists(user_path):
        print("Path not found. Please edit the 'user_path' variable in the code.")
    else:
        main(
            os.path.join(user_path, "Cell0278_SubBG1k.tif"),
            os.path.join(user_path, "Cell0279_SubBG1k.tif"),
            os.path.join(user_path, "C2-Composite.tif"),
            os.path.join(user_path, "Cell0278.pptx")
        )