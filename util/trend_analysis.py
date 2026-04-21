import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

def analyze_profile_trend(norm_r, norm_g):
    """
    Classification Logic
    Returns: (classification_string, correlation_coefficient)
    """
    # --- NEW: Gaussian Fit for Sigma ---
    try:
        # Define Gaussian function: y = A * exp(-x^2 / (2*sigma^2)) + B
        # Fixed center at x=0
        def gaussian_half(x, sigma, bg):
            return (norm_r[0] - bg) * np.exp(-x**2 / (2 * sigma**2)) + bg

        x_data = np.arange(len(norm_r))
        # Initial guess: sigma=2, bg=min(norm_r)
        p0 = [2.0, np.min(norm_r)]
        
        # Bounds: sigma > 0, bg >= 0
        popt, _ = curve_fit(gaussian_half, x_data, norm_r, p0=p0, bounds=([0.1, 0], [np.inf, np.inf]), maxfev=1000)
        sigma = popt[0]
        
    except Exception:
        sigma = 0 # Fit failed
    
    sigma_safe = max(sigma, 2) # Ensure sigma is positive and reasonable
    # -----------------------------------
    # 1. Calculate Correlation
    end = min(int(3 * sigma_safe) + 1, len(norm_g))
    abbe_limit = min(int(2.2884682938 * sigma_safe) + 2, len(norm_g))

    r_value = np.corrcoef(norm_r[:end], norm_g[:end])[0, 1]

    if np.isnan(r_value): r_value = 0

    # 2. Find Peaks
    prominence = np.ptp(norm_g) / 4
    if prominence <= 0: prominence = 0.001 
        
    peaks, _ = signal.find_peaks(norm_g, prominence=prominence)

    # 3. Logic Tree
    classification = "No trend"
    partition_coefficient_r = max(norm_r[:abbe_limit]) / min(norm_r)
    partition_coefficient_g = 1

    if (r_value >= 0.5 or max(norm_g) - max(norm_g[:1]) < 0.8 * prominence or (norm_g[0] > norm_g[1] and norm_g[1] > norm_g[2] and norm_g[0] - norm_g[3] > 1.6 * prominence)) and norm_g[0] > np.median(norm_g):
        classification = "Colocalized"
        partition_coefficient_g = max(norm_g[:abbe_limit]) / min(norm_g)
    elif (r_value <= -0.5 or min(norm_g[:1]) == min(norm_g)) and np.median(norm_g) - min(norm_g[:2]) >= 2.4 * prominence:
        classification = "Anticolocalized"
    elif np.any((peaks >= 1) & (peaks <= abbe_limit)) or max(norm_g) - max(norm_g[:abbe_limit]) <= prominence:
        classification = "Around"
    elif r_value <= -0.5:
        classification = "Anticolocalized"

    
    # --- CHANGED: Return tuple (Class, R-Value) ---
    return classification, r_value, sigma, partition_coefficient_r, partition_coefficient_g