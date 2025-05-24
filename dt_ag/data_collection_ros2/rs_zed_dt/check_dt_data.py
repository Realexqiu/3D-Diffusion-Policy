#!/usr/bin/env python3
"""
DenseTact + RGB Gallery Viewer
==============================
This enhanced script lets you inspect DenseTact *difference* frames (or raw
frames) **and**, side-by-side in a *second* OpenCV window, the *corresponding* RGB
images captured at the same timesteps.

Key additions
-------------
1. **Automatic RGB-dataset discovery** via `find_rgb_dataset` - no manual
   tweaking needed if your file contains keys like `zed_color_images`,
   `rs_color_images`, `*_rgb`, or anything with `color`/`rgb` in its name.
2. **Frame alignment logic** - when showing DenseTact *differences* the viewer
   aligns the RGB gallery to the *second* frame of each pair so that what you
   see in the colour window matches the timestamp of the *current* tactile
   difference.
3. **Independent gallery window** - the RGB grid is created with the same
   thumbnail size / grid layout as the DenseTact grid, but in its own OpenCV
   window. Controls (`Esc` or `q`) close *both* windows simultaneously.
4. **Adaptive grid layout** - automatically adjusts the grid layout to
   better fit your screen, using more horizontal space.
5. **Proper RGB display** - correctly displays RGB images in their original
   color format.

Feel free to change `FRAME_THUMB_SIZE` or the screen-size hints
(1920x1080) to suit your monitor.
"""

import os
import h5py
import numpy as np
import cv2
import glob
import math

###############################################################
# Helper utilities
###############################################################

def find_rgb_dataset(f):
    """Return the first 4-D dataset that *looks* like RGB images.

    Priority order tries keys that contain (case-insensitive):
    1. "zed_color"  2. "rs_color"  3. "rgb"  4. "color".
    """
    priority = ["zed_color_images", "rs_color_images"]
    for p in priority:
        for k in f.keys():
            if p in k.lower():
                ds = f[k]
                if isinstance(ds, h5py.Dataset) and len(ds.shape) == 4 and ds.shape[-1] in (3, 4):
                    return k
    
    # If priority keys not found, try generic rgb/color keywords
    for k in f.keys():
        if "rgb" in k.lower() or "color" in k.lower():
            ds = f[k]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) == 4 and ds.shape[-1] in (3, 4):
                return k
                
    return None


def load_rgb_frames(f, rgb_key, start_idx, end_idx):
    """Load RGB frames `[start_idx, end_idx)` and keep original RGB format."""
    frames = []
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, f[rgb_key].shape[0])
    
    for i in range(start_idx, end_idx):
        img = f[rgb_key][i]
        # ensure displayable type
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Keep the original RGB format - DON'T convert to BGR
        frames.append(img)
    return frames


def calculate_optimal_grid(n_frames, aspect_ratio=16/9):
    """
    Calculate optimal grid layout to maximize screen usage.
    
    Args:
        n_frames: Number of frames to display
        aspect_ratio: Screen aspect ratio (width/height), default 16:9
        
    Returns:
        (cols, rows): Optimal grid dimensions
    """
    # Start with a square-ish grid
    cols = int(math.sqrt(n_frames))
    rows = math.ceil(n_frames / cols)
    
    # Calculate the grid aspect ratio
    grid_ratio = cols / rows
    
    # Try to get closer to the screen aspect ratio
    while grid_ratio < aspect_ratio and cols < n_frames:
        cols += 1
        rows = math.ceil(n_frames / cols)
        grid_ratio = cols / rows
        
    return cols, rows


def create_grid_view(frames, sample_indices, thumb_size, window_name, screen_size=(1920, 1080)):
    """Create a grid mosaic of `frames` that maximizes screen usage."""
    # Calculate optimal grid layout based on screen aspect ratio
    screen_width, screen_height = screen_size
    screen_aspect = screen_width / screen_height
    
    # Get optimal grid dimensions
    cols, rows = calculate_optimal_grid(len(sample_indices), screen_aspect)
    
    print(f"Using grid layout: {cols} columns × {rows} rows (for {len(sample_indices)} frames)")
    
    # Create canvas with proper dimensions
    grid_h = rows * thumb_size[1]
    grid_w = cols * thumb_size[0]
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Place frames on canvas
    for i, idx in enumerate(sample_indices):
        r = i // cols
        c = i % cols
        
        # Ensure idx is within bounds
        if idx >= len(frames):
            continue
            
        frame = frames[idx]
        
        # Handle RGB to BGR conversion for OpenCV display
        if len(frame.shape) == 2:
            # Grayscale needs to be converted to BGR for display
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) > 2 and frame.shape[2] == 4:
            # RGBA needs to be converted to BGR for display
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) > 2 and frame.shape[2] == 3:
            # RGB needs to be converted to BGR for OpenCV display
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            disp_frame = frame
            
        # Resize frame to thumbnail size
        thumb = cv2.resize(disp_frame, thumb_size)
        
        # Add frame index label
        frame_idx_label = str(sample_indices[i])
        cv2.putText(thumb, frame_idx_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Place thumbnail on canvas
        y0 = r * thumb_size[1]
        x0 = c * thumb_size[0]
        canvas[y0 : y0 + thumb_size[1], x0 : x0 + thumb_size[0]] = thumb

    # Ensure we only create one window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, canvas)
    
    return cols, rows


def show_scaled_window(win_name, grid_w, grid_h, screen_w=1920, screen_h=1080):
    """Resize `win_name` so the mosaic fits inside `screen_*`."""
    # No need to create the window again, it's already created in create_grid_view
    
    # Calculate scale to fit within screen
    scale = min(screen_w / grid_w, screen_h / grid_h, 1.0)
    
    # Apply scale
    new_w, new_h = int(grid_w * scale), int(grid_h * scale)
    cv2.resizeWindow(win_name, new_w, new_h)

###############################################################
# Main CLI tool
###############################################################

def main():
    # ------------------------------------------------------------------
    # USER-CONFIGURABLE CONSTANTS
    # ------------------------------------------------------------------
    demo_data_dir = os.path.join(os.getcwd(), "demo_data")
    SENSOR_TO_DISPLAY = "left"          # "left" or "right" DenseTact sensor
    SHOW_DIFFERENCES = True             # True -> show frame-to-frame diffs
    START_FRAME, END_FRAME = 0, None    # inclusive range (END_FRAME=None -> all)

    DIFF_THRESHOLD = 10                  # pixels < threshold are nulled
    DIFF_SCALING = 5.0                  # brightness multiplier for diffs

    FRAME_THUMB_SIZE = (300, 300)       # gallery thumbnail size
    
    # Try to get screen resolution from the system
    try:
        import tkinter as tk
        root = tk.Tk()
        SCREEN_WIDTH = root.winfo_screenwidth()
        SCREEN_HEIGHT = root.winfo_screenheight()
        root.destroy()
        print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    except:
        # Default to 1080p if we can't get the screen resolution
        SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
        print(f"Using default screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    hdf5_files = sorted(glob.glob(os.path.join(demo_data_dir, "*.hdf5")))
    if not hdf5_files:
        print(f"No HDF5 files found in {demo_data_dir}")
        return
    print("Available HDF5 files:")
    for i, fp in enumerate(hdf5_files):
        print(f"  {i}: {os.path.basename(fp)}")
    try:
        file_idx = int(input("\nChoose a file index: "))
        if not 0 <= file_idx < len(hdf5_files):
            print("Invalid index")
            return
    except ValueError:
        print("Please enter a number")
        return

    hdf5_path = hdf5_files[file_idx]
    print(f"\nOpening {hdf5_path}\n")

    # ------------------------------------------------------------------
    # HDF5 loading & dataset discovery
    # ------------------------------------------------------------------
    with h5py.File(hdf5_path, "r") as f:
        # Locate DenseTact datasets
        dt_left, dt_right = None, None
        for k in f.keys():
            if "dt_1" in k or "dt_left" in k:
                dt_left = k
            elif "dt_2" in k or "dt_right" in k:
                dt_right = k

        sensor_key = dt_left if SENSOR_TO_DISPLAY == "left" else dt_right
        if sensor_key is None:
            print(f"Desired DenseTact sensor '{SENSOR_TO_DISPLAY}' not found!")
            print("Available keys:", list(f.keys()))
            return
        sensor_name = "Left" if SENSOR_TO_DISPLAY == "left" else "Right"

        # Get total frames and validate frame range
        total_frames = f[sensor_key].shape[0]
        print(f"Total frames available: {total_frames}")
        
        if END_FRAME is None:
            END_FRAME = total_frames
        
        # Ensure valid frame range
        START_FRAME = max(0, min(START_FRAME, total_frames - 2))  # Need at least 2 frames for diff
        END_FRAME = max(START_FRAME + 1, min(END_FRAME, total_frames))
        
        raw_n = END_FRAME - START_FRAME
        print(f"Using frames {START_FRAME}-{END_FRAME - 1} ({raw_n} frames)\n")

        # ------------------------------------------------------------------
        # Load DenseTact frames and compute differences (if requested)
        # ------------------------------------------------------------------
        print(f"Loading {raw_n} raw frames...")
        raw_frames = []
        for i in range(START_FRAME, END_FRAME):
            raw_frames.append(f[sensor_key][i].copy())  # Use .copy() to ensure we have our own data

        if SHOW_DIFFERENCES:
            print("Computing frame differences...")
            diff_frames = []
            for i in range(1, len(raw_frames)):
                prev = raw_frames[i - 1].astype(np.float32)
                curr = raw_frames[i].astype(np.float32)
                diff = np.abs(curr - prev)
                diff[diff < DIFF_THRESHOLD] = 0
                diff = np.clip(diff * DIFF_SCALING, 0, 255).astype(np.uint8)
                
                # Apply color map for better visualization
                if diff.ndim == 2 or (diff.shape[2] == 1 if len(diff.shape) > 2 else False):
                    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                else:
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    diff = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
                    
                diff_frames.append(diff)
            tactile_frames = diff_frames
        else:
            tactile_frames = raw_frames

        N = len(tactile_frames)
        print(f"Prepared {N} tactile {'difference ' if SHOW_DIFFERENCES else ''}frames")

        # ------------------------------------------------------------------
        # Load matching RGB frames (if available)
        # ------------------------------------------------------------------
        rgb_key = find_rgb_dataset(f)
        rgb_frames = None
        if rgb_key:
            # Align RGB frames with tactile frames
            # If showing differences, align with the second frame of each pair
            rgb_start = START_FRAME + (1 if SHOW_DIFFERENCES else 0)
            rgb_end = rgb_start + N
            
            print(f"Loading RGB frames from {rgb_key} (indices {rgb_start}-{rgb_end-1})...")
            try:
                rgb_frames = load_rgb_frames(f, rgb_key, rgb_start, rgb_end)
                print(f"Found RGB dataset '{rgb_key}' - loaded {len(rgb_frames)} frames\n")
            except Exception as e:
                print(f"Error loading RGB frames: {e}")
                rgb_frames = None
        else:
            print("No RGB dataset found - skipping colour gallery\n")

        # ------------------------------------------------------------------
        # Create galleries (DenseTact + RGB) - separate windows with optimal layout
        # ------------------------------------------------------------------
        # Use all frame indices for display
        indices = list(range(len(tactile_frames)))

        # DenseTact window
        frame_offset = (1 if SHOW_DIFFERENCES else 0)
        dt_win = f"DenseTact {sensor_name} {'Diffs' if SHOW_DIFFERENCES else 'Images'}" \
            f" (frames {START_FRAME + frame_offset}–{START_FRAME + frame_offset + N - 1})"
            
        print(f"Creating DenseTact grid view with {len(indices)} frames...")
        dt_cols, dt_rows = create_grid_view(
            tactile_frames, 
            indices, 
            FRAME_THUMB_SIZE, 
            dt_win,
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        
        grid_width = dt_cols * FRAME_THUMB_SIZE[0]
        grid_height = dt_rows * FRAME_THUMB_SIZE[1]
        show_scaled_window(dt_win, grid_width, grid_height, SCREEN_WIDTH, SCREEN_HEIGHT)

        # RGB window (if RGB frames are available)
        if rgb_frames:
            rgb_win = f"RGB {rgb_key} (frames {START_FRAME + frame_offset}-{START_FRAME + frame_offset + N - 1})"
            print(f"Creating RGB grid view with {len(indices)} frames...")
            rgb_cols, rgb_rows = create_grid_view(
                rgb_frames, 
                indices, 
                FRAME_THUMB_SIZE, 
                rgb_win,
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            
            rgb_grid_width = rgb_cols * FRAME_THUMB_SIZE[0]
            rgb_grid_height = rgb_rows * FRAME_THUMB_SIZE[1]
            show_scaled_window(rgb_win, rgb_grid_width, rgb_grid_height, SCREEN_WIDTH, SCREEN_HEIGHT)

        # ------------------------------------------------------------------
        # Simple key-loop - close on Esc / q
        # ------------------------------------------------------------------
        print("\nPress Esc or 'q' in *either* window to quit.")
        
        # Process any pending events to make sure windows appear
        cv2.waitKey(1)
        
        while True:
            k = cv2.waitKey(100) & 0xFF  # Reduced wait time for responsiveness
            if k in (27, ord("q")):
                break
                
            # Check if main window was closed
            try:
                if cv2.getWindowProperty(dt_win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break  # Window might not exist
                
            # Check if RGB window was closed (if it exists)
            if rgb_frames:
                try:
                    if cv2.getWindowProperty(rgb_win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    pass  # RGB window might not exist
        
        # Make sure all windows are properly destroyed  
        cv2.destroyAllWindows()
        # Process events to ensure windows are actually destroyed
        cv2.waitKey(1)


if __name__ == "__main__":
    main()