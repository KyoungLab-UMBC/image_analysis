import numpy as np

def normalize_minmax(img, high_bright = True):
    """
    Normalize array to 0-1 range using the 99th percentile.
    Uses 1/100 pixel sampling for speed.
    """
    img = img.astype(np.float32)
    
    # Sample 1 out of every 100 pixels
    sample = img.ravel()[::100]
    
    # Fallback: If image is smaller than 100 pixels, use the whole image
    if sample.size == 0:
        sample = img
    
    if high_bright:    
        min_v = np.min(sample)
        # CHANGED: Use 99th percentile instead of max to ignore hot pixels
        max_v = np.percentile(sample, 99.4)
    else:
        min_v = np.percentile(sample, 1.4)
        max_v = np.max(sample)
    
    # Safety check: if image is flat (or percentile equals min)
    if max_v - min_v <= 0:
        # Try falling back to actual max if percentile failed to find range
        max_v = np.max(sample)
        if max_v - min_v <= 0:
            return np.zeros_like(img)
        
    # Normalize
    normalized = (img - min_v) / (max_v - min_v)
    
    # Clip result to 0-1 range
    # This clamps the top 1% brightest pixels to 1.0
    return np.clip(normalized, 0, 1)