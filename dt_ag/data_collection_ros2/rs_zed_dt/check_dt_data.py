#!/usr/bin/env python3
import os
import h5py
import numpy as np
import cv2
import glob
import math

def main():
    # Hardcoded data directory path
    demo_data_dir = os.path.join(os.getcwd(), "demo_data")
    
    # HARDCODED CONTROL: Which sensor to display - set to "left" or "right"
    SENSOR_TO_DISPLAY = "left"  # Change to "right" to display right sensor images
    
    # HARDCODED CONTROL: Whether to show raw images or differences between consecutive frames
    SHOW_DIFFERENCES = True
    
    # FRAME RANGE OF INTEREST (modify these to focus on your frames of interest)
    START_FRAME = 30   # First frame to analyze
    END_FRAME = 50  # Last frame to analyze (None = use all frames)
    
    # DIFFERENCE VISUALIZATION SETTINGS
    # Threshold for difference detection (higher = less sensitive)
    DIFF_THRESHOLD = 5  # Values below this threshold will be set to 0
    
    # Scaling factor for differences (higher = brighter visualization)
    DIFF_SCALING = 3.0  # Multiply differences by this factor for better visibility
    
    # DISPLAY SETTINGS FOR FOCUSED VIEW
    # Size of each frame in the grid view (larger for better detail)
    FRAME_THUMB_SIZE = (300, 300)  # Much larger thumbnails for detailed view
    
    # Grid layout: number of columns 
    GRID_COLS = 6  # Fewer columns to allow for larger images
    
    # Get list of all HDF5 files in the directory
    hdf5_files = sorted(glob.glob(os.path.join(demo_data_dir, "*.hdf5")))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {demo_data_dir}")
        return
    
    print("Available HDF5 files:")
    for i, file_path in enumerate(hdf5_files):
        print(f"{i}: {os.path.basename(file_path)}")
    
    # Let user choose a file
    try:
        file_idx = int(input("\nEnter the file index to view: "))
        if file_idx < 0 or file_idx >= len(hdf5_files):
            print(f"Invalid file index. Please choose between 0 and {len(hdf5_files)-1}")
            return
    except ValueError:
        print("Please enter a valid number")
        return
        
    hdf5_path = hdf5_files[file_idx]
    print(f"\nUsing file: {hdf5_path}")
    
    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Print available datasets
        print("\nAvailable datasets in this file:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"- {key}: {f[key].shape}")
            else:
                print(f"- {key} (group)")
        
        # Find DenseTact keys
        dt_left_key = None
        dt_right_key = None
        
        for key in f.keys():
            if "dt_1" in key or "dt_left" in key:
                dt_left_key = key
            elif "dt_2" in key or "dt_right" in key:
                dt_right_key = key
        
        # Determine which sensor to use based on the hardcoded variable
        if SENSOR_TO_DISPLAY == "left":
            if not dt_left_key:
                print("No DenseTact left data found in this file!")
                return
            sensor_key = dt_left_key
            sensor_name = "Left"
        else:  # SENSOR_TO_DISPLAY == "right"
            if not dt_right_key:
                print("No DenseTact right data found in this file!")
                return
            sensor_key = dt_right_key
            sensor_name = "Right"
            
        # Determine total frames
        if len(f[sensor_key].shape) > 0:
            total_frames = f[sensor_key].shape[0]
        else:
            print(f"No valid frames found for DenseTact {sensor_name} sensor")
            return
        
        print(f"Total frames available: {total_frames}")
        
        # Set frame range if END_FRAME is not specified
        if END_FRAME is None:
            END_FRAME = total_frames
        
        # Validate frame range
        START_FRAME = max(0, min(START_FRAME, total_frames - 2))
        END_FRAME = max(START_FRAME + 1, min(END_FRAME, total_frames))
        actual_frame_count = END_FRAME - START_FRAME
        
        print(f"Focusing on frame range: {START_FRAME} to {END_FRAME-1} ({actual_frame_count} frames)")
        
        # We need at least 2 frames to compute differences
        if SHOW_DIFFERENCES and actual_frame_count < 2:
            print("At least 2 frames are required to compute differences")
            return
            
        # Update frame input dialog to allow setting range
        frame_range = input(f"\nEnter frame range as 'start-end' (e.g., '10-50') or press Enter to use {START_FRAME}-{END_FRAME-1}: ")
        if frame_range.strip():
            try:
                parts = frame_range.split('-')
                if len(parts) == 2:
                    START_FRAME = max(0, min(int(parts[0]), total_frames - 2))
                    END_FRAME = max(START_FRAME + 1, min(int(parts[1]) + 1, total_frames))
                    actual_frame_count = END_FRAME - START_FRAME
                    print(f"Updated frame range: {START_FRAME} to {END_FRAME-1} ({actual_frame_count} frames)")
                else:
                    print("Invalid range format. Using default range.")
            except ValueError:
                print("Invalid numbers in range. Using default range.")
        
        # Preload frames for the selected sensor in the specified range
        frames = []
        for i in range(START_FRAME, END_FRAME):
            frames.append(f[sensor_key][i])
            
        # Compute differences between consecutive frames if requested
        diff_frames = []
        if SHOW_DIFFERENCES:
            print("Computing frame differences...")
            max_diff_value = 0  # Track the maximum difference across all frames
            
            # First pass: compute differences and find maximum difference value
            temp_diffs = []
            for i in range(1, len(frames)):
                # Convert to float for subtraction to avoid uint8 underflow
                prev_frame = frames[i-1].astype(np.float32)
                curr_frame = frames[i].astype(np.float32)
                
                # Compute absolute difference
                diff = np.abs(curr_frame - prev_frame)
                
                # Apply threshold to remove noise
                diff[diff < DIFF_THRESHOLD] = 0
                
                # Track maximum difference for later normalization
                frame_max = np.max(diff)
                if frame_max > max_diff_value:
                    max_diff_value = frame_max
                    
                temp_diffs.append(diff)
            
            # Check if we found any significant differences
            if max_diff_value <= 0:
                print("Warning: No significant differences found between frames (all frames are identical)")
                max_diff_value = 1  # Avoid division by zero
            else:
                print(f"Maximum detected difference: {max_diff_value}")
            
            # Second pass: normalize differences based on maximum value and apply scaling
            for diff in temp_diffs:
                # Apply scaling factor to make differences more visible
                scaled_diff = diff * DIFF_SCALING
                
                # Clip to 0-255 range
                scaled_diff = np.clip(scaled_diff, 0, 255).astype(np.uint8)
                
                # Add colormap for better visualization (COLORMAP_JET shows differences in color)
                if len(diff.shape) == 2 or diff.shape[2] == 1:  # Grayscale
                    colored_diff = cv2.applyColorMap(scaled_diff, cv2.COLORMAP_JET)
                else:  # Already has color channels
                    # For color images, convert to grayscale first, then apply colormap
                    gray_diff = cv2.cvtColor(scaled_diff, cv2.COLOR_BGR2GRAY)
                    colored_diff = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
                
                diff_frames.append(colored_diff)
                
            # Use difference frames instead of raw frames
            frames = diff_frames
            total_frames = len(frames)  # One less than original since we compute differences
            print(f"Computed {total_frames} difference frames")
            
            # Count frames with no significant differences
            zero_diff_count = sum(1 for diff in temp_diffs if np.max(diff) < DIFF_THRESHOLD)
            if zero_diff_count > 0:
                print(f"Note: {zero_diff_count} frames ({zero_diff_count/total_frames*100:.1f}%) show no significant differences")
        
        # No downsampling - use all frames in the selected range
        sample_indices = list(range(total_frames))
        print(f"Showing all {total_frames} frames in high-resolution grid view")
            
        # Create grid layout
        window_name = f"DenseTact {sensor_name} {'Differences' if SHOW_DIFFERENCES else 'Images'} (Frames {START_FRAME}-{END_FRAME-1})"
        create_grid_view(frames, sample_indices, FRAME_THUMB_SIZE, window_name, GRID_COLS)
        
        # Create full-screen window with OpenCV window properties
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Initial resize to a reasonable size that fits most screens
        # This will be overridden if user toggles fullscreen
        screen_width, screen_height = 1920, 1080  # Assumed screen size
        grid_width = GRID_COLS * FRAME_THUMB_SIZE[0]
        grid_height = math.ceil(total_frames / GRID_COLS) * FRAME_THUMB_SIZE[1]
        
        # Scale to fit screen if needed
        scale = min(screen_width / grid_width, screen_height / grid_height)
        if scale < 1:
            new_width, new_height = int(grid_width * scale), int(grid_height * scale)
            cv2.resizeWindow(window_name, new_width, new_height)
        
        print("\nHigh-resolution grid view created. Press any key in the image window to close.")
        print("Press 'f' to toggle fullscreen, 'Esc' to close.")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if fullscreen == 0:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 0)
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        # After showing grid view, offer to show individual frames
        show_individual = input("\nWould you like to view individual frames? (y/n): ").lower()
        if show_individual == 'y':
            view_individual_frames(f, frames, total_frames, sensor_name, SHOW_DIFFERENCES, START_FRAME)


def create_grid_view(frames, sample_indices, thumb_size, window_name, cols):
    """Create a grid view of frames or frame differences"""
    rows = math.ceil(len(sample_indices) / cols)
    
    # Create grid canvas
    grid_height = rows * thumb_size[1]
    grid_width = cols * thumb_size[0]
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place frames in grid
    for i, frame_idx in enumerate(sample_indices):
        row = i // cols
        col = i % cols
        
        frame = frames[frame_idx]
        
        # Ensure frame is in RGB format for display
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
        # Resize frame to thumbnail size
        thumb = cv2.resize(frame, thumb_size)
        
        # Add frame number as text overlay
        cv2.putText(thumb, f"{frame_idx}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Place in grid
        y_start = row * thumb_size[1]
        y_end = y_start + thumb_size[1]
        x_start = col * thumb_size[0]
        x_end = x_start + thumb_size[0]
        
        grid[y_start:y_end, x_start:x_end] = thumb
    
    # Display grid
    cv2.imshow(window_name, grid)


def view_individual_frames(f, frames, total_frames, sensor_name, show_differences, start_frame_offset=0):
    """Allow interactive viewing of individual frames"""
    current_frame = 0
    
    print("\nIndividual Frame Viewer")
    print("Controls:")
    print("  → or d: Next frame")
    print("  ← or a: Previous frame")
    print("  j: Jump to specific frame")
    print("  + or = : Zoom in")
    print("  - : Zoom out")
    print("  f: Toggle fullscreen")
    print("  q or Esc: Quit")
    
    frame_type = "Difference" if show_differences else "Frame"
    window_title = f"{frame_type} - DenseTact {sensor_name}"
    
    # Create a resizable window for individual frame viewing
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    # Set initial window size to a reasonable value (adjust as needed)
    cv2.resizeWindow(window_title, 1024, 768)  # Larger window for detailed view
    
    # Track zoom level
    zoom_level = 1.0
    
    while True:
        # Display frame number (adjusted for range offset)
        actual_frame_num = start_frame_offset + current_frame
        next_frame_num = start_frame_offset + current_frame + 1
        
        print(f"\nViewing {frame_type} {current_frame}/{total_frames-1} (Original frame: {actual_frame_num})")
        
        # Show frame for current index
        sensor_image = frames[current_frame].copy()
        
        # Apply current zoom level
        if zoom_level != 1.0:
            h, w = sensor_image.shape[:2]
            new_h, new_w = int(h * zoom_level), int(w * zoom_level)
            sensor_image = cv2.resize(sensor_image, (new_w, new_h))
        
        title = window_title
        if show_differences:
            title += f" (Diff between frames {actual_frame_num} and {next_frame_num})"
        else:
            title += f" - Frame {actual_frame_num}"
            
        cv2.setWindowTitle(window_title, title)
        cv2.imshow(window_title, sensor_image)
        
        # Display additional data for the current frame if available and if using raw frames
        orig_frame_idx = actual_frame_num
        if show_differences:
            # If viewing differences, the original data is from the next frame
            orig_frame_idx += 1
            
        if "pose" in f and orig_frame_idx < len(f["pose"]):
            pose = f["pose"][orig_frame_idx]
            print(f"Robot pose: {pose}")
        
        if "gripper" in f and orig_frame_idx < len(f["gripper"]):
            gripper = f["gripper"][orig_frame_idx]
            print(f"Gripper: {gripper}")
        
        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF
        
        # Navigation controls
        if key == ord('q') or key == 27:  # 'q' or Esc key
            break
        elif key == ord('d') or key == 83:  # 'd' or right arrow
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord('a') or key == 81:  # 'a' or left arrow
            current_frame = max(current_frame - 1, 0)
        elif key == ord('j'):
            try:
                jump_to = int(input(f"Enter frame number (0-{total_frames-1}): "))
                if 0 <= jump_to < total_frames:
                    current_frame = jump_to
                else:
                    print(f"Invalid frame number. Must be between 0 and {total_frames-1}")
            except ValueError:
                print("Please enter a valid number")
        elif key == ord('+') or key == ord('='):  # Zoom in
            zoom_level *= 1.2
            print(f"Zoom level: {zoom_level:.1f}x")
        elif key == ord('-'):  # Zoom out
            zoom_level = max(0.1, zoom_level / 1.2)
            print(f"Zoom level: {zoom_level:.1f}x")
        elif key == ord('f'):  # Toggle fullscreen
            fullscreen = cv2.getWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN)
            if fullscreen == 0:
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, 0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()