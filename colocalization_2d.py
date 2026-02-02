import matplotlib
# Force Agg backend to prevent GUI errors
matplotlib.use('Agg') 

# Plotting imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import tifffile
from skimage import measure
from pptx import Presentation
from pptx.util import Inches
import io
import warnings
import sys
import os
import concurrent.futures
from PIL import Image 
import pandas as pd

# --- CUSTOM MODULE IMPORTS ---
import background_correction as bg_tools
import trend_analysis as trend_tools

# Initialize ImageJ once at the start
bg_tools.init_imagej()

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- HELPER FUNCTIONS ---

def create_plot_image(red_data, green_data, x_axis, obj_id):
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvasAgg(fig)
    
    ax1 = fig.add_subplot(111)
    
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
        ax1.set_title(f"Object: {obj_id}")
    
    fig.tight_layout()
    
    img_stream = io.BytesIO()
    canvas.print_png(img_stream)
    img_stream.seek(0)
    return img_stream

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
    
    img = Image.fromarray(rgb)
    img_stream = io.BytesIO()
    img.save(img_stream, format='PNG')
    img_stream.seek(0)
    return img_stream

def process_single_object(data_package):
    (obj_id, cy, cx, area, crop_r, crop_g, crop_mask) = data_package
    
    h, w = crop_r.shape
    center_y, center_x = h // 2, w // 2
    
    # --- 1. Background Estimation (IMPORTED) ---
    bg_r = bg_tools.estimate_background_rolling_ball(crop_r, radius=10)
    bg_r[bg_r == 0] = 1
    norm_img_r = crop_r.astype(float) / bg_r
    
    bg_g = bg_tools.estimate_background_rolling_ball(crop_g, radius=10)
    bg_g[bg_g == 0] = 1
    norm_img_g = crop_g.astype(float) / bg_g
    
    # --- 2. Radial Profile ---
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
    
    # --- 3. Classification (IMPORTED) ---
    size_class = "small glucosome" if area <= 14 else "large glucosome"
    trend_class = trend_tools.analyze_profile_trend(final_r, final_g)
    
    # --- 4. Plotting ---
    x_vals = np.arange(0, 21)
    
    plot_bytes = create_plot_image(final_r, final_g, x_vals, obj_id).getvalue()
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

    results = []
    
    # ThreadPool
    max_workers = min(32, os.cpu_count() + 4)
    print(f"Processing with ThreadPool ({max_workers} workers)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, res in enumerate(executor.map(process_single_object, tasks)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f"Analyzed {i+1}/{len(tasks)}")

    # --- PPT GENERATION ---
    print("Generating PowerPoint Report...")
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
            print(f"Saved PPT: {output_pptx}")
            break
        except PermissionError:
            input(f"File {output_pptx} is OPEN. Close it and press ENTER...")

    # --- EXCEL GENERATION (CUSTOM FORMAT) ---
    print("Generating Excel Report...")
    output_xlsx = output_pptx.replace(".pptx", ".xlsx")
    
    categories = [
        ("Colocalized", "small glucosome"),
        ("Colocalized", "large glucosome"),
        ("Around", "small glucosome"),
        ("Around", "large glucosome"),
        ("Anticolocalized", "small glucosome"),
        ("Anticolocalized", "large glucosome"),
        ("No trend", "small glucosome"),
        ("No trend", "large glucosome")
    ]

    try:
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            for trend, size in categories:
                subset = [r for r in results if r['trend_class'] == trend and r['size_class'] == size]
                
                header_row = ["Normalized GAPDH"] + list(range(21)) + ["", "Normalized PFKL"] + list(range(21))
                data_rows = [header_row]
                
                for item in subset:
                    row_data = [item['id']] + list(item['norm_g']) + ["", item['id']] + list(item['norm_r'])
                    data_rows.append(row_data)
                
                if len(data_rows) > 1:
                    df = pd.DataFrame(data_rows)
                else:
                    df = pd.DataFrame([header_row]) 

                sheet_name = f"{trend} {size}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"Saved Excel: {output_xlsx}")
        
    except Exception as e:
        print(f"Failed to save Excel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Point this to your specific files
    user_path = r"F:\20251217 PFKL-mCherry_EGFP-GAPDH - High Salt Conc - 37 degree - WideField\Plate 1 - 180 mM\cell4\1_AddSalt45min"
    
    if not os.path.exists(user_path):
        print("Path not found. Please edit the 'user_path' variable in the code.")
    else:
        main(
            os.path.join(user_path, "Cell0110_SubBG1k.tif"),
            os.path.join(user_path, "Cell0111_SubBG1k.tif"),
            os.path.join(user_path, "C2-Composite.tif"),
            os.path.join(user_path, "Cell0110.pptx")
        )