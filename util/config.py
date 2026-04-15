from pathlib import Path

#Input
USER_PATH = Path(r"F:\20250507 PFKL-mCherry_Hylight_MitotrackerDeepred - High Salt Conc - 37 degree - WideField\Plate 1 - 140 mM NaCl\Cell1")

R_NAME = "PFKL"
R_RAW = USER_PATH / "Cell0000.tif"

G_NAME = "Pyruvate"
G_PATH = USER_PATH / "Cell0049_SubBG1k.tif"

MITO_RAW = USER_PATH / "Cell0003.tif"

# Analysis Switches
DEBUG_MODE = False   # Whether to save intermediate masks and outputs for debugging purposes
GENERATE_SLIDES = True # Whether to generate PowerPoint slides with results (can be time-consuming)
MITO_ANALYSIS = True # Whether to perform mitochondrial analysis (requires MITO_PATH)
BIOSENSOR = True     # True - biosensor analysis; False - enzyme analysis (e.g., PKM2)
QUEEN37C = False     # True - use Queen37C biosensor; False - use Hylight biosensor

# Below are processed automatically based on the above inputs
R_PATH = R_RAW #USER_PATH / f"{R_RAW.stem}_SubBG1k.tif"

if BIOSENSOR:
    if QUEEN37C:
        G_NAME = "ATP"
        G_PATH = USER_PATH / "Queen37C_ratiox10000.tif"
    else:
        G_NAME = "FBP"
        G_PATH = USER_PATH / "Hylight_ratiox1000.tif"
