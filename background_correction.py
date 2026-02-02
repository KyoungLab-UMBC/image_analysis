import os
import sys
import numpy as np
from scyjava import jimport
import imagej

# ---- Initialize ImageJ Global Variables ----
ij = None
IJ = None

def init_imagej():
    """
    Initializes ImageJ if it hasn't been initialized yet.
    """
    global ij, IJ
    if ij is not None:
        return # Already initialized

    print("Initializing ImageJ...")
    try:
        ij = imagej.init('2.14.0', mode='headless')
        print("Success! ImageJ version:", ij.getVersion())
        IJ = jimport('ij.IJ')
    except Exception as e:
        print(f"Initialization Failed: {e}")
        sys.exit()

def estimate_background_rolling_ball(img, radius=10):
    """
    Estimates background using ImageJ's Native 'Subtract Background'.
    """
    # Ensure ImageJ is initialized before running
    if ij is None:
        init_imagej()

    # 1. Convert NumPy -> ImagePlus
    imp_source = ij.py.to_imageplus(img)
    
    # 2. DUPLICATE to avoid in-place modification
    imp_background = imp_source.duplicate()

    # 3. Run Rolling Ball
    options = f"rolling={radius} create"
    IJ.run(imp_background, "Subtract Background...", options)

    # 4. Convert back to NumPy
    background = ij.py.from_java(imp_background)
    
    res_np = np.array(background)
    if res_np.ndim > 2: res_np = res_np.squeeze()
        
    return res_np.astype(img.dtype)