#!/usr/bin/env python3
import os, glob
import h5py
import numpy as np
import cv2

def main():
    # Hardcoded parameters
    hdf5_dir = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_dt/3d_strawberry_dt_50_hdf5"
    crop_x = 400   # Crop box top-left X
    crop_y = 100   # Crop box top-left Y
    crop_w = 1400   # Crop box width
    crop_h = 2000   # Crop box height
    
    # Maximum display width (adjust as needed for your monitor)
    max_display_width = 1280
    max_display_height = 720

    # 1) find first file
    files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    if not files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return
    hf = h5py.File(files[5], "r")
    if "zed_color_images" not in hf:
        print("Dataset 'zed_color_images' not found in", files[0])
        return

    # 2) grab first frame
    frame = hf["zed_color_images"][19]  # shape (H, W, 3), BGR uint8
    hf.close()
    
    # Print original dimensions
    print(f"Original image dimensions: {frame.shape}")

    # 3) Convert BGR to RGB for visualization
    orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4) Apply crop
    cropped = orig[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
    print(f"Cropped image dimensions: {cropped.shape}")

    # 5) Resize cropped image to fit on screen
    crop_aspect = cropped.shape[1] / cropped.shape[0]
    scale_factor = min(max_display_width / cropped.shape[1], max_display_height / cropped.shape[0])
    
    # Only resize if the image is too large
    if scale_factor < 1:
        new_width = int(cropped.shape[1] * scale_factor)
        new_height = int(cropped.shape[0] * scale_factor)
        crop_resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Resized cropped image to: {crop_resized.shape}")
    else:
        crop_resized = cropped
        print("Cropped image already fits on screen")

    # 6) Show (convert back to BGR for cv2.imshow)
    crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
    cv2.imshow("Cropped Image", crop_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()