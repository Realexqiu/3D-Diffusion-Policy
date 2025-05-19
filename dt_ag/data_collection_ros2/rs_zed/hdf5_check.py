#!/usr/bin/env python3
import os
import h5py
import numpy as np
import cv2
import glob

# Standard display size for all windows

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

def main():
    # Hardcoded demo data directory path
    demo_data_dir = os.path.join(os.getcwd(), "demo_data")
    
    # Get list of all HDF5 files in the directory
    hdf5_files = sorted(glob.glob(os.path.join(demo_data_dir, "*.hdf5")))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {demo_data_dir}")
        return
    
    # Choose the most recent file by default
    file_idx = len(hdf5_files) - 1
    hdf5_path = hdf5_files[file_idx]
    print(f"Using file: {hdf5_path}")
    
    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Print available datasets
        print("Available datasets in this file:")
        for key in f.keys():
            print(f"- {key}: {f[key].shape}")
        
        # Function to resize image to standard display size
        def resize_for_display(image):
            return cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Function to visualize depth as colormap
        def visualize_depth(image):
            if len(image.shape) != 2:  # Must be (H, W)
                print(f"Depth image has unexpected shape {image.shape}, skipping.")
                return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            
            # Check image type
            is_float = np.issubdtype(image.dtype, np.floating)
            
            if is_float:
                # Handle floating point (meters) depth data
                # Filter out invalid values (NaN, inf, zeros)
                valid_mask = np.logical_and(np.isfinite(image), image > 0)
                
                if not np.any(valid_mask):
                    print("No valid depth data found")
                    return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                
                # Use valid values for normalization
                valid_depths = image[valid_mask]
                d_min, d_max = np.min(valid_depths), np.max(valid_depths)
                print(f"Depth range: {d_min:.3f} to {d_max:.3f} meters")
            else:
                # Handle integer (millimeters) depth data
                valid_mask = image > 0
                
                if not np.any(valid_mask):
                    print("No valid depth data found")
                    return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                
                # Use valid values for normalization
                valid_depths = image[valid_mask]
                d_min, d_max = np.min(valid_depths), np.max(valid_depths)
                print(f"Depth range: {d_min} to {d_max} millimeters")
            
            if d_min == d_max:
                d_min, d_max = 0, 1  # Avoid divide-by-zero
            
            # Create normalized copy
            depth_normalized = np.zeros_like(image, dtype=np.float32)
            depth_normalized[valid_mask] = 255 * (image[valid_mask] - d_min) / (d_max - d_min)
            
            # Convert to uint8 and apply colormap
            depth_viz = depth_normalized.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            
            # For areas with no valid depth, make them black
            depth_colormap[~valid_mask] = [0, 0, 0]
            
            # Resize to standard display size
            return resize_for_display(depth_colormap)
        
        # Get frame count
        total_frames = 0
        for key in f.keys():
            if len(f[key].shape) >= 1 and f[key].shape[0] > 1:
                total_frames = f[key].shape[0]
                break
                
        if total_frames == 0:
            print("Could not determine frame count from datasets")
            return
            
        print(f"Total frames: {total_frames}")
        
        # We'll display first and last frames
        first_frame_idx = 0
        last_frame_idx = total_frames - 1
        
        # Organize windows with cv2.moveWindow
        window_positions = {
            # First frame windows
            "First-RS-Color": (0, 0),
            "First-RS-Depth": (DISPLAY_WIDTH, 0),
            "First-ZED-Color": (0, DISPLAY_HEIGHT),
            "First-ZED-Depth": (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            
            # Last frame windows
            "Last-RS-Color": (3*DISPLAY_WIDTH, 0),
            "Last-RS-Depth": (4*DISPLAY_WIDTH, 0),
            "Last-ZED-Color": (3*DISPLAY_WIDTH, DISPLAY_HEIGHT),
            "Last-ZED-Depth": (4*DISPLAY_WIDTH, DISPLAY_HEIGHT)
        }
        
        # First frame display
        print("\n--- First Frame (Index 0) ---")
        
        # Display RealSense color and depth (first frame)
        if "rs_color_images" in f and len(f["rs_color_images"]) > 0:
            rs_first_color = f["rs_color_images"][first_frame_idx]
            rs_color_display = resize_for_display(cv2.cvtColor(rs_first_color, cv2.COLOR_RGB2BGR))
            cv2.imshow("First-RS-Color", rs_color_display)
            cv2.moveWindow("First-RS-Color", *window_positions["First-RS-Color"])
        
        if "rs_depth_images" in f and len(f["rs_depth_images"]) > 0:
            rs_first_depth = f["rs_depth_images"][first_frame_idx]
            rs_depth_display = visualize_depth(rs_first_depth)
            cv2.imshow("First-RS-Depth", rs_depth_display)
            cv2.moveWindow("First-RS-Depth", *window_positions["First-RS-Depth"])
        
        # Display ZED color and depth (first frame)
        if "zed_color_images" in f and len(f["zed_color_images"]) > 0:
            zed_first_color = f["zed_color_images"][first_frame_idx]
            zed_color_display = resize_for_display(zed_first_color)
            cv2.imshow("First-ZED-Color", zed_color_display)
            cv2.moveWindow("First-ZED-Color", *window_positions["First-ZED-Color"])
        
        if "zed_depth_images" in f and len(f["zed_depth_images"]) > 0:
            zed_first_depth = f["zed_depth_images"][first_frame_idx]
            zed_depth_display = visualize_depth(zed_first_depth)
            cv2.imshow("First-ZED-Depth", zed_depth_display)
            cv2.moveWindow("First-ZED-Depth", *window_positions["First-ZED-Depth"])
        
        # Last frame display
        print("\n--- Last Frame (Index {}) ---".format(last_frame_idx))
        
        # Display RealSense color and depth (last frame)
        if "rs_color_images" in f and len(f["rs_color_images"]) > 0:
            rs_last_color = f["rs_color_images"][last_frame_idx]
            rs_color_display = resize_for_display(cv2.cvtColor(rs_last_color, cv2.COLOR_RGB2BGR))
            cv2.imshow("Last-RS-Color", rs_color_display)
            cv2.moveWindow("Last-RS-Color", *window_positions["Last-RS-Color"])
        
        if "rs_depth_images" in f and len(f["rs_depth_images"]) > 0:
            rs_last_depth = f["rs_depth_images"][last_frame_idx]
            rs_depth_display = visualize_depth(rs_last_depth)
            cv2.imshow("Last-RS-Depth", rs_depth_display)
            cv2.moveWindow("Last-RS-Depth", *window_positions["Last-RS-Depth"])
        
        # Display ZED color and depth (last frame)
        if "zed_color_images" in f and len(f["zed_color_images"]) > 0:
            zed_last_color = f["zed_color_images"][last_frame_idx]
            zed_color_display = resize_for_display(zed_last_color)
            cv2.imshow("Last-ZED-Color", zed_color_display)
            cv2.moveWindow("Last-ZED-Color", *window_positions["Last-ZED-Color"])
        
        if "zed_depth_images" in f and len(f["zed_depth_images"]) > 0:
            zed_last_depth = f["zed_depth_images"][last_frame_idx]
            zed_depth_display = visualize_depth(zed_last_depth)
            cv2.imshow("Last-ZED-Depth", zed_depth_display)
            cv2.moveWindow("Last-ZED-Depth", *window_positions["Last-ZED-Depth"])
        
        print("\nPress any key to close all windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()