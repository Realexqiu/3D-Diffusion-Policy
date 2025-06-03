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
INPUT_ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/10_hz_baseline_100_zarr")

# Output Zarr dataset path
OUTPUT_ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/10_hz_baseline_100_zarr_no_crop")

# Resize settings
ENABLE_RESIZE = True  # Set to False to keep original resolution
TARGET_WIDTH = 220    # Target width for resizing
TARGET_HEIGHT = 180   # Target height for resizing

# Center crop settings
ENABLE_CENTER_CROP_RS = False  # Set to False to disable center cropping
RS_CROP_WIDTH = 850      # Target width for center crop
RS_CROP_HEIGHT = 420     # Target height for center crop
ENABLE_CENTER_CROP_ZED = False
ZED_CROP_WIDTH = 320
ZED_CROP_HEIGHT = 360

# Off-center crop settings (set to None for center crop, or specify offset)
# Positive values move crop towards bottom-right, negative towards top-left
RS_CROP_OFFSET_X = None    # Horizontal offset from center (None for center crop)
RS_CROP_OFFSET_Y = None    # Vertical offset from center (None for center crop)
ZED_CROP_OFFSET_X = -300   # For ZED camera
ZED_CROP_OFFSET_Y = None   # For ZED camera

# Color jitter settings
ENABLE_COLOR_JITTER = False  # Set to False to disable color jitter
COLOR_JITTER_BRIGHTNESS = 0.15  # Brightness variation (0.0 = no change)
COLOR_JITTER_CONTRAST = 0.15    # Contrast variation (0.0 = no change)
COLOR_JITTER_SATURATION = 0.15  # Saturation variation (0.0 = no change)
COLOR_JITTER_HUE = 0.0          # Hue variation (0.0 = no change)

# Debug mode - set to True to process only first episode
DEBUGGING = False

CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def apply_color_jitter(rgb_frames: np.ndarray, 
                      brightness: float = 0.0,
                      contrast: float = 0.0, 
                      saturation: float = 0.0,
                      hue: float = 0.0) -> np.ndarray:
    """
    Apply color jitter to RGB frames.
    
    Args:
        rgb_frames: Input frames in (T, C, H, W) or (T, H, W, C) format
        brightness: Brightness factor range (0.0 = no change)
        contrast: Contrast factor range (0.0 = no change)
        saturation: Saturation factor range (0.0 = no change)
        hue: Hue shift range (0.0 = no change)
    
    Returns:
        Color jittered frames in same format as input
    """
    if brightness == 0.0 and contrast == 0.0 and saturation == 0.0 and hue == 0.0:
        return rgb_frames
    
    # Convert to HWC format for processing
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1, 3):
        # already (T, H, W, C)
        hwc = rgb_frames
        input_format = 'hwc'
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1, 3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0, 2, 3, 1)
        input_format = 'chw'
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")
    
    T, H, W, C = hwc.shape
    jittered_hwc = np.zeros_like(hwc)
    
    for t in range(T):
        frame = hwc[t].astype(np.float32)
        
        # Apply brightness jitter
        if brightness > 0.0:
            brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
            frame = frame * brightness_factor
        
        # Apply contrast jitter
        if contrast > 0.0:
            contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
            mean = np.mean(frame)
            frame = (frame - mean) * contrast_factor + mean
        
        # Convert to HSV for saturation and hue adjustments
        if saturation > 0.0 or hue > 0.0:
            # Convert RGB to HSV
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply saturation jitter
            if saturation > 0.0:
                saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            
            # Apply hue jitter
            if hue > 0.0:
                hue_shift = np.random.uniform(-hue, hue) * 180  # Convert to OpenCV hue range
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Convert back to RGB
            hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Clip values to valid range
        jittered_hwc[t] = np.clip(frame, 0, 255).astype(hwc.dtype)
    
    # Convert back to original format
    if input_format == 'chw':
        return jittered_hwc.transpose(0, 3, 1, 2)
    else:
        return jittered_hwc

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

def crop_rgb_frames(rgb_frames: np.ndarray, 
                   crop_size: Tuple[int,int],
                   offset_x: Optional[int] = None,
                   offset_y: Optional[int] = None) -> np.ndarray:
    """
    Crop RGB frames to target size with optional offset, returning (T, C, H, W).
    
    Args:
        rgb_frames: Input frames
        crop_size: (width, height) of crop
        offset_x: Horizontal offset from center (None for center crop)
        offset_y: Vertical offset from center (None for center crop)

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
    
    # Calculate crop coordinates
    center_y = H // 2
    center_x = W // 2
    
    # Apply offsets (default to center crop if None)
    if offset_x is None:
        start_x = center_x - crop_w // 2
    else:
        start_x = center_x - crop_w // 2 + offset_x
        
    if offset_y is None:
        start_y = center_y - crop_h // 2
    else:
        start_y = center_y - crop_h // 2 + offset_y
    
    # Ensure crop stays within image bounds
    start_x = max(0, min(start_x, W - crop_w))
    start_y = max(0, min(start_y, H - crop_h))
    
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
    required_arrays = ["rs_side_rgb", "zed_rgb", "pose", "action"]
    
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
            input_episode = input_root[episode_name]
            if not check_required_arrays(input_episode, episode_name):
                continue
            
            # Load required data
            rs_side_rgb = input_episode["rs_side_rgb"][:]
            zed_rgb     = input_episode["zed_rgb"][:]
            pose        = input_episode["pose"][:]
            action      = input_episode["action"][:]
            T = rs_side_rgb.shape[0]

            # Apply cropping if enabled
            if ENABLE_CENTER_CROP_RS:
                crop_type = "center" if RS_CROP_OFFSET_X is None and RS_CROP_OFFSET_Y is None else "off-center"
                CONSOLE.log(f"[blue]Applying {crop_type} crop to RS frames from {rs_side_rgb.shape[1:3]} to ({RS_CROP_HEIGHT}, {RS_CROP_WIDTH})")
                rs_side_rgb = crop_rgb_frames(rs_side_rgb, (RS_CROP_WIDTH, RS_CROP_HEIGHT),
                                              RS_CROP_OFFSET_X, RS_CROP_OFFSET_Y)

            if ENABLE_CENTER_CROP_ZED:
                crop_type = "center" if ZED_CROP_OFFSET_X is None and ZED_CROP_OFFSET_Y is None else "off-center"
                CONSOLE.log(f"[blue]Applying {crop_type} crop to ZED frames from {zed_rgb.shape[1:3]} to ({ZED_CROP_HEIGHT}, {ZED_CROP_WIDTH})")
                zed_rgb = crop_rgb_frames(zed_rgb, (ZED_CROP_WIDTH, ZED_CROP_HEIGHT),
                                          ZED_CROP_OFFSET_X, ZED_CROP_OFFSET_Y)

            # Apply color jitter if enabled
            if ENABLE_COLOR_JITTER:
                CONSOLE.log(f"[blue]Applying color jitter (brightness={COLOR_JITTER_BRIGHTNESS}, contrast={COLOR_JITTER_CONTRAST}, saturation={COLOR_JITTER_SATURATION}, hue={COLOR_JITTER_HUE})")
                rs_side_rgb = apply_color_jitter(rs_side_rgb,
                                                 COLOR_JITTER_BRIGHTNESS,
                                                 COLOR_JITTER_CONTRAST,
                                                 COLOR_JITTER_SATURATION,
                                                 COLOR_JITTER_HUE)
                zed_rgb = apply_color_jitter(zed_rgb,
                                             COLOR_JITTER_BRIGHTNESS,
                                             COLOR_JITTER_CONTRAST,
                                             COLOR_JITTER_SATURATION,
                                             COLOR_JITTER_HUE)

            # Resize RGB frames if enabled
            if ENABLE_RESIZE:
                CONSOLE.log(f"[blue]Resizing RGB frames from {rs_side_rgb.shape[1:3]} to ({TARGET_HEIGHT}, {TARGET_WIDTH})")
                rs_side_rgb = resize_rgb_frames(rs_side_rgb, (TARGET_WIDTH, TARGET_HEIGHT))
                zed_rgb = resize_rgb_frames(zed_rgb, (TARGET_WIDTH, TARGET_HEIGHT))

            # Create output episode group
            output_episode = output_root.create_group(episode_name)
            output_episode.array("rs_side_rgb", rs_side_rgb, dtype=np.uint8, chunks=(1, *rs_side_rgb.shape[1:]))
            output_episode.array("zed_rgb",     zed_rgb,     dtype=np.uint8, chunks=(1, *zed_rgb.shape[1:]))
            output_episode.array("pose",   pose,   dtype=np.float32)
            output_episode.array("action", action, dtype=np.float32)
            
            # Copy episode attributes
            if hasattr(input_episode, 'attrs'):
                for attr_name, attr_value in input_episode.attrs.items():
                    output_episode.attrs[attr_name] = attr_value
            
            output_episode.attrs["length"] = T
            successful_episodes += 1
            CONSOLE.log(f"[green]✓ Completed {episode_name}")
            
        except Exception as e:
            CONSOLE.log(f"[red]Error processing {episode_name}: {e}")
            if episode_name in output_root:
                del output_root[episode_name]
            continue

    # ────────────────────────────────────────────────────────────
    #  Store one-shot preprocessing metadata at the Zarr **root**
    # ────────────────────────────────────────────────────────────
    preprocess_meta = {
        "resize": {
            "enabled": ENABLE_RESIZE,
            "target": [TARGET_WIDTH, TARGET_HEIGHT],          # [W, H]
        },
        "crop": {
            "rs_side": {
                "enabled": ENABLE_CENTER_CROP_RS,
                "size":   [RS_CROP_WIDTH, RS_CROP_HEIGHT],    # [W, H]
                "offset": [RS_CROP_OFFSET_X, RS_CROP_OFFSET_Y],
            },
            "zed": {
                "enabled": ENABLE_CENTER_CROP_ZED,
                "size":   [ZED_CROP_WIDTH, ZED_CROP_HEIGHT],
                "offset": [ZED_CROP_OFFSET_X, ZED_CROP_OFFSET_Y],
            },
        },
        "color_jitter": {
            "enabled":    ENABLE_COLOR_JITTER,
            "brightness": COLOR_JITTER_BRIGHTNESS,
            "contrast":   COLOR_JITTER_CONTRAST,
            "saturation": COLOR_JITTER_SATURATION,
            "hue":        COLOR_JITTER_HUE,
        },
    }
    output_root.attrs["preprocess"] = preprocess_meta
    # ────────────────────────────────────────────────────────────

    # Final summary
    CONSOLE.log(f"[green]Processing complete!")
    CONSOLE.log(f"[green]Successfully processed: {successful_episodes}/{len(episode_names)} episodes")
    CONSOLE.log(f"[green]Output dataset saved to: {OUTPUT_ZARR_DIR}")

    # Transformation log (unchanged)
    transformations = []
    if ENABLE_CENTER_CROP_RS:
        crop_type = "center" if RS_CROP_OFFSET_X is None and RS_CROP_OFFSET_Y is None else f"off-center ({RS_CROP_OFFSET_X}, {RS_CROP_OFFSET_Y})"
        transformations.append(f"RS cropped ({crop_type}) to: {RS_CROP_HEIGHT}x{RS_CROP_WIDTH}")
    if ENABLE_CENTER_CROP_ZED:
        crop_type = "center" if ZED_CROP_OFFSET_X is None and ZED_CROP_OFFSET_Y is None else f"off-center ({ZED_CROP_OFFSET_X}, {ZED_CROP_OFFSET_Y})"
        transformations.append(f"ZED cropped ({crop_type}) to: {ZED_CROP_HEIGHT}x{ZED_CROP_WIDTH}")
    if ENABLE_COLOR_JITTER:
        transformations.append(f"Color jitter applied (brightness={COLOR_JITTER_BRIGHTNESS}, contrast={COLOR_JITTER_CONTRAST})")
    if ENABLE_RESIZE:
        transformations.append(f"Resized to: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    
    if transformations:
        CONSOLE.log(f"[green]Transformations applied: {', '.join(transformations)}")
    else:
        CONSOLE.log(f"[green]RGB frames kept at original resolution")
    
    if successful_episodes > 0:
        sample_episode = output_root[episode_names[0]]
        CONSOLE.log(f"[cyan]Dataset info:")
        CONSOLE.log(f"[cyan]  - Episodes: {successful_episodes}")
        CONSOLE.log(f"[cyan]  - RS RGB shape: {sample_episode['rs_side_rgb'].shape}")
        CONSOLE.log(f"[cyan]  - ZED RGB shape: {sample_episode['zed_rgb'].shape}")
        CONSOLE.log(f"[cyan]  - Pose shape: {sample_episode['pose'].shape}")
        CONSOLE.log(f"[cyan]  - Action shape: {sample_episode['action'].shape}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()