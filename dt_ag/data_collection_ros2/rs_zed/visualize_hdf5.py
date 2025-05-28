#!/usr/bin/env python3
import os
import h5py
import numpy as np
import cv2
import glob

# Standard display size for all windows
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

def resize_for_display(image):
    return cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

def visualize_depth(image):
    if image.ndim != 2:
        print(f"Depth image has unexpected shape {image.shape}, skipping.")
        return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    is_float = np.issubdtype(image.dtype, np.floating)
    if is_float:
        valid = np.logical_and(np.isfinite(image), image > 0)
        if not np.any(valid):
            print("No valid depth data found")
            return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        vals = image[valid]
        d_min, d_max = vals.min(), vals.max()
        print(f"Depth range: {d_min:.3f} to {d_max:.3f} meters")
    else:
        valid = image > 0
        if not np.any(valid):
            print("No valid depth data found")
            return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        vals = image[valid]
        d_min, d_max = vals.min(), vals.max()
        print(f"Depth range: {d_min} to {d_max} millimeters")

    if d_min == d_max:
        d_min, d_max = 0, 1

    norm = np.zeros_like(image, dtype=np.float32)
    norm[valid] = 255 * (image[valid] - d_min) / (d_max - d_min)
    depth_viz = norm.astype(np.uint8)
    cmap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    cmap[~valid] = [0, 0, 0]
    return resize_for_display(cmap)

def show_episode(hdf5_path):
    print(f"\n=== Episode: {hdf5_path} ===")
    with h5py.File(hdf5_path, "r") as f:
        print("Datasets:")
        for k in f.keys():
            print(f"  {k}: {f[k].shape}")

        # determine frame count
        total = 0
        for k in f.keys():
            if f[k].ndim >= 1 and f[k].shape[0] > 1:
                total = f[k].shape[0]
                break
        if total == 0:
            print("Could not determine frame count, skipping.")
            return

        first_idx, last_idx = 0, total - 1

        # window positions
        w, h = DISPLAY_WIDTH, DISPLAY_HEIGHT
        pos = {
            "RS-C-1": (0,   0),
            "RS-D-1": (w,   0),
            "ZED-C-1": (0,   h),
            "ZED-D-1": (w,   h),
            "RS-C-2": (3*w, 0),
            "RS-D-2": (4*w, 0),
            "ZED-C-2": (3*w, h),
            "ZED-D-2": (4*w, h),
        }

        def display_frame(idx, suffix):
            # RealSense color
            if "rs_color_images" in f:
                img = f["rs_color_images"][idx]
                disp = resize_for_display(img)
                name = f"RS-C-{suffix}"
                cv2.imshow(name, disp)
                cv2.moveWindow(name, *pos[name])
            # RealSense depth
            if "rs_depth_images" in f:
                img = f["rs_depth_images"][idx]
                disp = visualize_depth(img)
                name = f"RS-D-{suffix}"
                cv2.imshow(name, disp)
                cv2.moveWindow(name, *pos[name])
            # ZED color
            if "zed_color_images" in f:
                img = f["zed_color_images"][idx]
                disp = resize_for_display(img)
                name = f"ZED-C-{suffix}"
                cv2.imshow(name, disp)
                cv2.moveWindow(name, *pos[name])
            # ZED depth
            if "zed_depth_images" in f:
                img = f["zed_depth_images"][idx]
                disp = visualize_depth(img)
                name = f"ZED-D-{suffix}"
                cv2.imshow(name, disp)
                cv2.moveWindow(name, *pos[name])

        print(f"\n-- Showing first frame (idx {first_idx}) --")
        display_frame(first_idx, "1")
        print(f"\n-- Showing last frame (idx {last_idx}) --")
        display_frame(last_idx, "2")

        print("\nPress any key to continue to next episode...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    demo_data_dir = os.path.join(os.getcwd(), "1")
    hdf5_files = sorted(glob.glob(os.path.join(demo_data_dir, "*.hdf5")))
    if not hdf5_files:
        print(f"No HDF5 files found in {demo_data_dir}")
        return

    for path in hdf5_files:
        show_episode(path)

if __name__ == "__main__":
    main()
