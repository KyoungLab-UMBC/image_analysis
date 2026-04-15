import numpy as np
import matplotlib.pyplot as plt
import io
import vtk
from vtk.util import numpy_support

def create_dimetric_thumbnail_matplot(crop_r, crop_g):
    """
    Creates a 3D Volumetric Thumbnail.
    - Black Background
    - Alpha Scales with Intensity (Max 0.5) to prevent occlusion
    - Additive Blending (Red + Green = Yellow)
    - Custom Contrast (Median to Peak)
    """
    fig = plt.figure(figsize=(4, 4), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    s_start, s_end = 4, 17
    sub_r = crop_r[s_start:s_end, s_start:s_end, s_start:s_end]
    sub_g = crop_g[s_start:s_end, s_start:s_end, s_start:s_end]
    
    cz, cy, cx = 6, 6, 6 

    # --- NORMALIZE FUNCTION ---
    def get_normalized_channel(arr, is_red):
        v_min = np.median(arr)
        
        if is_red:
            center_val = arr[cz, cy, cx]
            v_max = center_val if center_val > v_min else np.max(arr)
            # Boost Red Exposure slightly to make mid-tones visible
            boost = 3.0 
        else:
            v_max = np.max(arr)
            boost = 1.0

        if v_max <= v_min: v_max = v_min + 1.0
            
        # Normalize
        arr_norm = (arr - v_min) / (v_max - v_min)
        arr_norm = arr_norm * boost
        return np.clip(arr_norm, 0, 1)

    norm_r = get_normalized_channel(sub_r, is_red=True)
    norm_g = get_normalized_channel(sub_g, is_red=False)

    # --- COMBINE & PLOT ---
    # 1. Calculate Combined Intensity (Max of R or G at each point)
    # This determines how "important" the pixel is.
    max_intensity = np.maximum(norm_r, norm_g)

    # 2. Threshold
    # We remove pixels that are too dark to matter (cleaner view)
    mask = max_intensity > 0.1 
    
    if np.any(mask):
        z, y, x = np.where(mask)
        
        colors = np.zeros((len(z), 4))
        colors[:, 0] = norm_r[mask] # Red
        colors[:, 1] = norm_g[mask] # Green
        colors[:, 2] = 0.0          # Blue
        
        # --- DYNAMIC ALPHA CORRECTION ---
        # Instead of fixed 0.5, we scale alpha by intensity.
        # Brightest pixel (1.0) -> 0.5 Alpha
        # Dim pixel (0.2)       -> 0.1 Alpha (Transparent)
        # This allows you to "see through" the dark parts.
        intensities = max_intensity[mask]
        colors[:, 3] = np.clip(intensities * 0.5, 0.05, 0.5)
        
        # Plot
        ax.scatter(x, y, z, c=colors, s=35, depthshade=False, 
                   marker='s', linewidth=0, edgecolors='none', antialiased=False)

    ax.view_init(elev=25, azim=45)
    
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=100, facecolor='black')
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

def create_dimetric_thumbnails_VTK(crop_r, crop_g):
    """
    Creates TWO 3D Volume Renderings (View 1 and View 2 rotated 90 deg).
    Mixes Red and Green channels into a single RGBA texture.
    
    UPDATES:
    1. Box Gradient: Edges fade to 0.2 (Chebyshev distance).
    2. Green Transparency: Green contributes only 30% to opacity to prevent blocking.
    """
    
    # --- 1. PRE-PROCESSING & DYNAMIC RANGE ---
    s_start, s_end = 4, 17
    sub_r = crop_r[s_start:s_end, s_start:s_end, s_start:s_end].astype(np.float32)
    sub_g = crop_g[s_start:s_end, s_start:s_end, s_start:s_end].astype(np.float32)
    
    d, h, w = sub_r.shape
    cz, cy, cx = d // 2, h // 2, w // 2

    # --- Red Dynamic Range: Min to Center Pixel ---
    min_r = np.min(sub_r)
    max_r = sub_r[cz, cy, cx] # Center pixel sets the Max
    
    if max_r <= min_r: 
        max_r = np.max(sub_r) 
        if max_r <= min_r: max_r = min_r + 1e-5

    norm_r = (sub_r - min_r) / (max_r - min_r)
    norm_r = np.clip(norm_r, 0.0, 1.0) 

    # --- Green Dynamic Range: Min to Max ---
    min_g = np.min(sub_g)
    max_g = np.max(sub_g)
    if max_g <= min_g: max_g = min_g + 1e-5
    
    norm_g = (sub_g - min_g) / (max_g - min_g)
    norm_g = np.clip(norm_g, 0.0, 1.0)

    # --- 2. MIX TO RGBA ---
    rgba_vol = np.zeros((d, h, w, 4), dtype=np.uint8)
    
    # Colors remain full strength (0-255)
    rgba_vol[..., 0] = (norm_r * 255).astype(np.uint8) # R
    rgba_vol[..., 1] = (norm_g * 255).astype(np.uint8) # G
    rgba_vol[..., 2] = 0    # B
    
    # --- 3. CALCULATE SMART ALPHA ---
    # A. Box Distance (Chebyshev) - Handles Cube Shape
    zz, yy, xx = np.indices((d, h, w))
    box_dist = np.maximum(np.abs(zz - cz), np.maximum(np.abs(yy - cy), np.abs(xx - cx)))
    
    max_dist = d // 2
    if max_dist == 0: max_dist = 1
    
    # Gradient: 1.0 at center -> 0.2 at box edges
    spatial_scale = 1.0 - (box_dist / max_dist) * 0.8
    spatial_scale = np.clip(spatial_scale, 0.2, 1.0)
    
    # B. Opacity Mixing (The Fix for Blocking)
    # Green contributes significantly LESS to opacity (0.3) than Red (1.0).
    # This makes Green "ghostly" and Red "solid".
    base_opacity = np.maximum(norm_r, norm_g * 0.3)
    
    # Final Alpha = Weighted Intensity * Spatial Gradient
    alpha_float = base_opacity * spatial_scale
    alpha_u8 = (alpha_float * 255).astype(np.uint8)
    
    rgba_vol[..., 3] = alpha_u8

    # --- 4. VTK RENDERING ---
    data_flat = rgba_vol.reshape(-1, 4)
    vtk_data = numpy_support.numpy_to_vtk(num_array=data_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    
    img = vtk.vtkImageData()
    img.SetDimensions(w, h, d)
    img.GetPointData().SetScalars(vtk_data)
    
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(img)
    
    vol_prop = vtk.vtkVolumeProperty()
    vol_prop.ShadeOn()
    vol_prop.SetInterpolationTypeToLinear()
    vol_prop.IndependentComponentsOff() 
    
    # Alpha 0-255 maps to Opacity 0.0-1.0
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(255, 1.0)
    vol_prop.SetScalarOpacity(opacity_func)
    
    vol_prop.SetAmbient(0.4)
    vol_prop.SetDiffuse(0.6)
    vol_prop.SetSpecular(0.2)
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(vol_prop)
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.AddVolume(volume)
    
    # Outline
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(img)
    mapper_outline = vtk.vtkPolyDataMapper()
    mapper_outline.SetInputConnection(outline.GetOutputPort())
    actor_outline = vtk.vtkActor()
    actor_outline.SetMapper(mapper_outline)
    actor_outline.GetProperty().SetColor(1, 1, 1)
    actor_outline.GetProperty().SetLineWidth(1.0)
    renderer.AddActor(actor_outline)

    # View 1
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(400, 400)
    
    camera = renderer.GetActiveCamera()
    camera.Elevation(20)
    camera.Azimuth(30)
    renderer.ResetCamera()
    camera.Zoom(1.3)
    
    render_window.Render()
    
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.WriteToMemoryOn()
    writer.Write()
    view1_bytes = bytes(writer.GetResult())

    # View 2 (Rotate 90)
    camera.Azimuth(90)
    render_window.Render()
    w2if.Modified()
    w2if.Update()
    writer.Write()
    view2_bytes = bytes(writer.GetResult())
    
    del volume, mapper, img, renderer, render_window, writer, w2if
    
    return view1_bytes, view2_bytes