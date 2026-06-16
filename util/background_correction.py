import sys
import numpy as np
import scyjava
import imagej
import itertools

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

def _restore_axis_order(arr, target_shape):
    if arr.shape == target_shape:
        return arr

    if arr.ndim != len(target_shape):
        return arr

    for perm in itertools.permutations(range(arr.ndim)):
        candidate = np.transpose(arr, perm)
        if candidate.shape == target_shape:
            return candidate

    return arr

def estimate_background_rolling_ball(img, radius=10, create_background=True, use_paraboloid=False, stack=False):
    if ij is None:
        init_imagej()

    imp_source = ij.py.to_imageplus(img)
    imp_copy = imp_source.duplicate()

    options_list = [f"rolling={radius}"]

    if create_background:
        options_list.append("create")

    if use_paraboloid:
        options_list.append("sliding")

    if stack:
        options_list.append("stack")

    options = " ".join(options_list)
    IJ.run(imp_copy, "Subtract Background...", options)

    result = ij.py.from_java(imp_copy)
    res_np = np.asarray(result)

    # Only squeeze if the input was not 3D
    if stack:
        res_np = _restore_axis_order(res_np, img.shape)
    else:
        res_np = np.squeeze(res_np)

    imp_source.close()
    imp_copy.close()

    return res_np.astype(img.dtype)

def force_garbage_collection():
    """
    Forces both Java and Python to clear unused memory.
    Crucial for preventing Heap Space errors in loops.
    """
    print("--- Cleaning Memory (Java + Python) ---")
    
    # 1. Force Python GC
    import gc
    gc.collect()

    # 2. Force Java GC (via ImageJ)
    if IJ is not None:
        # This triggers System.gc() internally in ImageJ
        IJ.freeMemory()