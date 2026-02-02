import numpy as np
from scipy import signal

def analyze_profile_trend(norm_r, norm_g):
    """
    Classification Logic
    """
    # 1. Calculate Correlation
    limit = min(len(norm_r), 10)
    if limit > 2:
        r_value = np.corrcoef(norm_r[:limit], norm_g[:limit])[0, 1]
    else:
        r_value = 0
    if np.isnan(r_value): r_value = 0

    # 2. Find Peaks
    g_segment = norm_g[:10]
    prominence = (np.max(g_segment) - np.min(g_segment)) / 5
    if prominence <= 0: prominence = 0.001 
        
    peaks, _ = signal.find_peaks(g_segment, prominence=prominence)

    # 3. Logic Tree
    if r_value >= 0.75 or (norm_g[0] > norm_g[1] and norm_g[1] > norm_g[2]):
        return "Colocalized"
    elif r_value <= -0.75 or norm_g[0] == min(norm_g) and np.median(norm_g) - norm_g[0] >= 3 * prominence:
        return "Anticolocalized"
    elif np.any((peaks >= 1) & (peaks <= 6)) or max(norm_g) - max(norm_g[1:4]) <= prominence:
        return "Around"
    else:
        return "No trend"