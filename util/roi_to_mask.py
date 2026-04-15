import numpy as np
from skimage import draw
from read_roi import read_roi_zip

def roi_to_mask(roi, image_shape):
    """
    Convert ROI data to a binary mask.
    Supports rectangle, polygon, freehand, traced, and oval.
    """
    mask = np.zeros(image_shape, dtype=bool)
    # Handle dict keys varying by ROI source (ImageJ vs others)
    roi_type = roi.get('type', '').lower() 
    
    try:
        if roi_type == 'rectangle':
            r_start = int(roi['top'])
            r_end = int(roi['top'] + roi['height'])
            c_start = int(roi['left'])
            c_end = int(roi['left'] + roi['width'])
            
            # Clip bounds
            r_start = max(0, r_start)
            c_start = max(0, c_start)
            r_end = min(image_shape[0], r_end)
            c_end = min(image_shape[1], c_end)
            
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

def load_roi_file(image_path):
    """
    Load ROI data from an ImageJ ROI zip file.
    Checks for a file named '<image_name>_rois.zip' first, 
    then falls back to a generic 'rois.zip' in the same directory.
    """
    potential_paths = [
        image_path.parent / f"{image_path.stem}_rois.zip",
        image_path.parent / "rois.zip"
    ]

    rois = None
    for p in potential_paths:
        if p.exists():
            try:
                # Convert Path object to string for read_roi_zip compatibility
                rois = read_roi_zip(str(p))
                print(f"  Found ROIs at: {p.name}")
                break
            except Exception as e:
                print(f"  Error reading {p.name}: {e}")

    if not rois:
        print(f"No valid ROI file found for {image_path.name}. Skipping.")
        return None
        
    # MUST return the loaded rois!
    return rois