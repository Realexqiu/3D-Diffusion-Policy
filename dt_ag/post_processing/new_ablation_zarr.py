#!/usr/bin/env python3
"""
Zarr dataset filter and resize script.

This script copies data from an existing Zarr dataset and creates a new
filtered dataset containing only desired data.

"""

# ────────────────────────────────────────────────────────────────
#  Standard imports
# ────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import zarr
from rich.console import Console
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
# Input Zarr dataset path
INPUT_ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline_zarr")

# Output Zarr dataset path
OUTPUT_ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline_zarr_2d")

# Resize settings
ENABLE_RESIZE = True  # Set to False to keep original resolution
TARGET_WIDTH = 480    # Target width for resizing
TARGET_HEIGHT = 360   # Target height for resizing

# Center crop settings
ENABLE_CENTER_CROP = True  # Set to False to disable center cropping
RS_CROP_WIDTH = 850      # Target width for center crop
RS_CROP_HEIGHT = 420     # Target height for center crop
ZED_CROP_WIDTH = 1400
ZED_CROP_HEIGHT = 1200

# Debug mode - set to True to process only first episode
DEBUGGING = False

CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def resize_rgb_frames(rgb_frames: np.ndarray, target_size: Tuple[int,int]) -> np.ndarray:
    """
    Resize RGB frames to target size, returning (T, C, H, W).

    Accepts either:
      - HWC: (T, H, W, C)
      - CHW: (T, C, H, W)
    """
    target_w, target_h = target_size

    # Step 1: get everything into HWC-per-frame
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1,3):
        # already (T, H, W, C)
        hwc = rgb_frames
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1,3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0,2,3,1)  # → (T, H, W, C)
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")

    T, H, W, C = hwc.shape
    out_hwc = np.zeros((T, target_h, target_w, C), dtype=hwc.dtype)

    # Step 2: resize each frame in HWC format
    for t in range(T):
        out_hwc[t] = cv2.resize(hwc[t],
                                (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR)

    # Step 3: convert back to CHW-per-frame
    return out_hwc.transpose(0, 3, 1, 2)    # → (T, C, H, W)

def center_crop_rgb_frames(rgb_frames: np.ndarray, crop_size: Tuple[int,int]) -> np.ndarray:
    """
    Center crop RGB frames to target size, returning (T, C, H, W).

    Accepts either:
      - HWC: (T, H, W, C)
      - CHW: (T, C, H, W)
    """
    crop_w, crop_h = crop_size

    # Step 1: get everything into HWC-per-frame
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1,3):
        # already (T, H, W, C)
        hwc = rgb_frames
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1,3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0,2,3,1)  # → (T, H, W, C)
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")

    T, H, W, C = hwc.shape
    
    # Check if crop size is valid
    if crop_h > H or crop_w > W:
        raise ValueError(f"Crop size ({crop_h}, {crop_w}) larger than image size ({H}, {W})")
    
    # Calculate center crop coordinates
    start_y = (H - crop_h) // 2
    start_x = (W - crop_w) // 2
    end_y = start_y + crop_h
    end_x = start_x + crop_w
    
    # Step 2: crop each frame
    cropped_hwc = hwc[:, start_y:end_y, start_x:end_x, :]
    
    # Step 3: convert back to CHW-per-frame
    return cropped_hwc.transpose(0, 3, 1, 2)    # → (T, C, H, W)

def get_episode_names(zarr_root) -> list:
    """Get sorted list of episode names from Zarr root."""
    episode_names = [key for key in zarr_root.keys() if key.startswith("episode_")]
    return sorted(episode_names)

def check_required_arrays(episode_group, episode_name: str) -> bool:
    """Check if episode has all required arrays."""
    required_arrays = ["rs_color_images", "pose", "action"]
    
    for array_name in required_arrays:
        if array_name not in episode_group:
            CONSOLE.log(f"[yellow]Episode {episode_name} missing {array_name}, skipping")
            return False
    
    return True

# ────────────────────────────────────────────────────────────────
#  Main conversion routine
# ────────────────────────────────────────────────────────────────
def main() -> None:
    """Main function to filter and resize Zarr dataset."""
    
    # Check if input directory exists
    if not INPUT_ZARR_DIR.exists():
        CONSOLE.log(f"[red]Input Zarr directory does not exist: {INPUT_ZARR_DIR}")
        return
    
    # Create output directory
    ensure_dir(OUTPUT_ZARR_DIR)
    
    try:
        # Open input Zarr dataset
        input_root = zarr.open(INPUT_ZARR_DIR, mode="r")
        CONSOLE.log(f"[green]Opened input Zarr dataset: {INPUT_ZARR_DIR}")
    except Exception as e:
        CONSOLE.log(f"[red]Error opening input Zarr dataset: {e}")
        return
    
    # Get episode names
    episode_names = get_episode_names(input_root)
    if not episode_names:
        CONSOLE.log("[red]No episodes found in input dataset")
        return
    
    if DEBUGGING:
        episode_names = episode_names[:1]
        CONSOLE.log(f"[yellow]Debug mode: processing only {episode_names[0]}")
    
    CONSOLE.log(f"[blue]Found {len(episode_names)} episodes to process")
    
    # Open output Zarr dataset
    output_root = zarr.open(OUTPUT_ZARR_DIR, mode="w")
    
    # Process each episode
    successful_episodes = 0
    
    for episode_name in tqdm(episode_names, desc="Processing episodes"):
        try:
            CONSOLE.log(f"[blue]Processing {episode_name}")
            
            # Get input episode group
            input_episode = input_root[episode_name]
            
            # Check if required arrays exist
            if not check_required_arrays(input_episode, episode_name):
                continue
            
            # Load required data
            rs_rgb = input_episode["rs_color_images"][:]
            zed_rgb = input_episode["zed_color_images"][:]
            pose = input_episode["pose"][:]
            action = input_episode["action"][:]
            
            # Get episode length
            T = rs_rgb.shape[0]

            # Apply center crop if enabled
            if ENABLE_CENTER_CROP:
                CONSOLE.log(f"[blue]Center cropping RGB frames from {rs_rgb.shape[1:3]} to ({RS_CROP_HEIGHT}, {RS_CROP_WIDTH})")
                rs_rgb = center_crop_rgb_frames(rs_rgb, (RS_CROP_WIDTH, RS_CROP_HEIGHT))
                zed_rgb = center_crop_rgb_frames(zed_rgb, (ZED_CROP_WIDTH, ZED_CROP_HEIGHT))

            # Resize RGB frames if enabled
            if ENABLE_RESIZE:
                CONSOLE.log(f"[blue]Resizing RGB frames from {rs_rgb.shape[1:3]} to ({TARGET_HEIGHT}, {TARGET_WIDTH})")
                rs_rgb = resize_rgb_frames(rs_rgb, (TARGET_WIDTH, TARGET_HEIGHT))
                zed_rgb = resize_rgb_frames(zed_rgb, (TARGET_WIDTH, TARGET_HEIGHT))

            # Create output episode group
            output_episode = output_root.create_group(episode_name)
            
            # Store filtered data
            output_episode.array("rs_color_images", rs_rgb, dtype=np.uint8, chunks=(1, *rs_rgb.shape[1:]))
            output_episode.array("zed_color_images", zed_rgb, dtype=np.uint8, chunks=(1, *zed_rgb.shape[1:]))
            output_episode.array("pose", pose, dtype=np.float32)
            output_episode.array("action", action, dtype=np.float32)
            
            # Copy episode attributes
            if hasattr(input_episode, 'attrs'):
                for attr_name, attr_value in input_episode.attrs.items():
                    output_episode.attrs[attr_name] = attr_value
            
            # Ensure length attribute is set
            output_episode.attrs["length"] = T
            
            if ENABLE_RESIZE:
                output_episode.attrs["original_resolution"] = f"{rs_rgb.shape[1]}x{rs_rgb.shape[2]}"
                output_episode.attrs["resized_to"] = f"{TARGET_HEIGHT}x{TARGET_WIDTH}"
            
            successful_episodes += 1
            CONSOLE.log(f"[green]✓ Completed {episode_name}")
            
        except Exception as e:
            CONSOLE.log(f"[red]Error processing {episode_name}: {e}")
            # Clean up partial episode data if it exists
            if episode_name in output_root:
                del output_root[episode_name]
            continue
    
    # Final summary
    CONSOLE.log(f"[green]Processing complete!")
    CONSOLE.log(f"[green]Successfully processed: {successful_episodes}/{len(episode_names)} episodes")
    CONSOLE.log(f"[green]Output dataset saved to: {OUTPUT_ZARR_DIR}")

    # Log transformations applied
    transformations = []
    if ENABLE_CENTER_CROP:
        transformations.append(f"Center cropped to: {RS_CROP_HEIGHT}x{RS_CROP_WIDTH}")
    if ENABLE_RESIZE:
        transformations.append(f"Resized to: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    
    if transformations:
        CONSOLE.log(f"[green]Transformations applied: {', '.join(transformations)}")
    else:
        CONSOLE.log(f"[green]RGB frames kept at original resolution")
    
    # Show dataset info
    if successful_episodes > 0:
        sample_episode = output_root[episode_names[0]]
        CONSOLE.log(f"[cyan]Dataset info:")
        CONSOLE.log(f"[cyan]  - Episodes: {successful_episodes}")
        CONSOLE.log(f"[cyan]  - RS RGB shape: {sample_episode['rs_color_images'].shape}")
        CONSOLE.log(f"[cyan]  - ZED RGB shape: {sample_episode['zed_color_images'].shape}")
        CONSOLE.log(f"[cyan]  - Pose shape: {sample_episode['pose'].shape}")
        CONSOLE.log(f"[cyan]  - Action shape: {sample_episode['action'].shape}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()