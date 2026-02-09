import sys
import numpy as np
import scyjava
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
        # Set Memory Limit for Java BEFORE init. 
        # Adjust '12g' to '8g' or '24g' depending on your PC's RAM.
        scyjava.config.add_option('-Xmx8g')
        ij = imagej.init('2.14.0', mode='headless')
        print("Success! ImageJ version:", ij.getVersion())
        IJ = scyjava.jimport('ij.IJ')
    except Exception as e:
        print(f"Initialization Failed: {e}")
        sys.exit()

def estimate_background_rolling_ball(img, radius=10, create_background=True, use_paraboloid=False):
    """
    Estimates or subtracts background using ImageJ's Native 'Subtract Background'.
    
    Parameters:
    - img: Input image (numpy array).
    - radius: Rolling ball radius.
    - create_background: If True, returns the estimated background. 
                         If False, returns the image with background subtracted.
    - use_paraboloid: If True, uses the 'Sliding Paraboloid' algorithm.
    """
    # Ensure ImageJ is initialized before running
    if ij is None:
        init_imagej()

    # 1. Convert NumPy -> ImagePlus
    imp_source = ij.py.to_imageplus(img)
    
    # 2. DUPLICATE to avoid in-place modification of the source
    # We call this imp_copy because it will become either the background OR the result
    imp_copy = imp_source.duplicate()

    # 3. Build Options String
    options_list = [f"rolling={radius}"]
    
    if create_background:
        options_list.append("create")
    
    if use_paraboloid:
        options_list.append("sliding")

    options = " ".join(options_list)

    # 4. Run Rolling Ball
    # Note: IJ.run modifies the image object in-place
    IJ.run(imp_copy, "Subtract Background...", options)

    # 5. Convert back to NumPy
    result = ij.py.from_java(imp_copy)
    
    res_np = np.array(result)
    if res_np.ndim > 2: res_np = res_np.squeeze()
        
    return res_np.astype(img.dtype)