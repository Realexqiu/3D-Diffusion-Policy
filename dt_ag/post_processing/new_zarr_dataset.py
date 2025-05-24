#!/usr/bin/env python3
"""
Zarr dataset filter and resize script.

This script copies data from an existing Zarr dataset and creates a new
filtered dataset containing only:
- rs_rgb (with optional resizing)
- agent_pos  
- action

All ZED camera data and depth information is excluded.
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
TARGET_WIDTH = 640    # Target width for resizing
TARGET_HEIGHT = 360   # Target height for resizing

# Debug mode - set to True to process only first episode
DEBUGGING = False

CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def resize_rgb_frames(rgb_frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize RGB frames to target size.
    
    Args:
        rgb_frames: Array of shape (T, H, W, C)
        target_size: (width, height) tuple
        
    Returns:
        Resized frames array of shape (T, target_height, target_width, C)
    """
    T, H, W, C = rgb_frames.shape
    target_width, target_height = target_size
    
    resized_frames = np.zeros((T, target_height, target_width, C), dtype=rgb_frames.dtype)
    
    for t in range(T):
        # OpenCV resize expects (width, height)
        resized_frame = cv2.resize(rgb_frames[t], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_frames[t] = resized_frame
    
    return resized_frames

def get_episode_names(zarr_root) -> list:
    """Get sorted list of episode names from Zarr root."""
    episode_names = [key for key in zarr_root.keys() if key.startswith("episode_")]
    return sorted(episode_names)

def check_required_arrays(episode_group, episode_name: str) -> bool:
    """Check if episode has all required arrays."""
    required_arrays = ["rs_rgb", "agent_pos", "action"]
    
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
            rs_rgb = input_episode["rs_rgb"][:]
            agent_pos = input_episode["agent_pos"][:]
            action = input_episode["action"][:]
            
            # Get episode length
            T = rs_rgb.shape[0]
            
            # Resize RGB frames if enabled
            if ENABLE_RESIZE:
                CONSOLE.log(f"[blue]Resizing RGB frames from {rs_rgb.shape[1:3]} to ({TARGET_HEIGHT}, {TARGET_WIDTH})")
                rs_rgb = resize_rgb_frames(rs_rgb, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # Create output episode group
            output_episode = output_root.create_group(episode_name)
            
            # Store filtered data
            output_episode.array("rs_rgb", rs_rgb, dtype=np.uint8, chunks=(1, *rs_rgb.shape[1:]))
            output_episode.array("agent_pos", agent_pos, dtype=np.float32)
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
    
    if ENABLE_RESIZE:
        CONSOLE.log(f"[green]RGB frames resized to: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    else:
        CONSOLE.log(f"[green]RGB frames kept at original resolution")
    
    # Show dataset info
    if successful_episodes > 0:
        sample_episode = output_root[episode_names[0]]
        CONSOLE.log(f"[cyan]Dataset info:")
        CONSOLE.log(f"[cyan]  - Episodes: {successful_episodes}")
        CONSOLE.log(f"[cyan]  - RS RGB shape: {sample_episode['rs_rgb'].shape}")
        CONSOLE.log(f"[cyan]  - Agent pos shape: {sample_episode['agent_pos'].shape}")
        CONSOLE.log(f"[cyan]  - Action shape: {sample_episode['action'].shape}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()