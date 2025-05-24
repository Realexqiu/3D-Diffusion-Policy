#!/usr/bin/env python3
"""
Zarr Dataset Inspector

Inspects a converted Zarr dataset to show the actual dimensions and data types
of each array so you can configure your task config properly.
"""

import zarr
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

CONSOLE = Console()

def inspect_zarr_dataset(zarr_path: Path, episode_idx: int = 0) -> None:
    """
    Inspect a single episode from the Zarr dataset to understand data shapes.
    
    Args:
        zarr_path: Path to the Zarr dataset
        episode_idx: Which episode to inspect (default: 0)
    """
    
    if not zarr_path.exists():
        CONSOLE.log(f"[red]Zarr path does not exist: {zarr_path}")
        return
    
    try:
        root = zarr.open(zarr_path, mode="r")
        
        # List all available episodes
        episodes = [key for key in root.keys() if key.startswith("episode_")]
        episodes.sort()
        
        if not episodes:
            CONSOLE.log("[red]No episodes found in Zarr dataset!")
            return
        
        CONSOLE.log(f"[green]Found {len(episodes)} episodes in dataset")
        CONSOLE.log(f"[blue]Episodes: {episodes[0]} to {episodes[-1]}")
        
        # Select episode to inspect
        if episode_idx >= len(episodes):
            CONSOLE.log(f"[yellow]Episode index {episode_idx} out of range, using episode 0")
            episode_idx = 0
        
        episode_name = episodes[episode_idx]
        episode_group = root[episode_name]
        
        CONSOLE.log(f"\n[bold blue]Inspecting {episode_name}:")
        
        # Create a table for the results
        table = Table(title=f"Data Shapes in {episode_name}")
        table.add_column("Array Name", style="cyan", no_wrap=True)
        table.add_column("Shape", style="magenta")
        table.add_column("Data Type", style="green")
        table.add_column("Min Value", style="yellow")
        table.add_column("Max Value", style="yellow")
        table.add_column("Notes", style="white")
        
        # Inspect each array in the episode
        for array_name in sorted(episode_group.keys()):
            array = episode_group[array_name]
            
            # Get basic info
            shape_str = str(array.shape)
            dtype_str = str(array.dtype)
            
            # Calculate min/max for small arrays or sample for large ones
            try:
                if array.size < 1000000:  # For reasonably sized arrays
                    min_val = np.min(array[:])
                    max_val = np.max(array[:])
                else:  # For very large arrays, sample
                    sample = array[:10] if len(array.shape) > 0 else array[:]
                    min_val = np.min(sample)
                    max_val = np.max(sample)
                min_str = f"{min_val:.3f}" if isinstance(min_val, (float, np.floating)) else str(min_val)
                max_str = f"{max_val:.3f}" if isinstance(max_val, (float, np.floating)) else str(max_val)
            except Exception as e:
                min_str = "N/A"
                max_str = "N/A"
            
            # Add notes based on array name and properties
            notes = ""
            if array_name in ["rs_rgb", "zed_rgb"]:
                notes = "RGB images (H, W, 3)"
            elif array_name in ["rs_depth", "zed_depth"]:
                notes = "Depth images (H, W)"
            elif array_name == "zed_pcd":
                notes = "Point clouds (T, N_points, 6) - [x,y,z,r,g,b]"
            elif array_name == "agent_pos":
                notes = "Robot poses (T, 7) - [x,y,z,qw,qx,qy,qz]"
            elif array_name == "action":
                notes = "Actions (T, 7) - pose deltas"
            
            table.add_row(array_name, shape_str, dtype_str, min_str, max_str, notes)
        
        CONSOLE.print(table)
        
        # Print episode attributes
        if hasattr(episode_group, 'attrs') and episode_group.attrs:
            CONSOLE.log(f"\n[bold blue]Episode Attributes:")
            for attr_name, attr_value in episode_group.attrs.items():
                CONSOLE.log(f"  {attr_name}: {attr_value}")
        
        # Generate suggested config
        generate_config_suggestion(episode_group)
        
    except Exception as e:
        CONSOLE.log(f"[red]Error inspecting Zarr dataset: {e}")
        import traceback
        traceback.print_exc()

def generate_config_suggestion(episode_group) -> None:
    """Generate a suggested configuration based on the actual data shapes."""
    
    CONSOLE.log(f"\n[bold green]Suggested Configuration Updates:")
    
    # Check RGB shapes (assuming rs_rgb and zed_rgb have same dimensions)
    if "rs_rgb" in episode_group and "zed_rgb" in episode_group:
        rs_shape = episode_group["rs_rgb"].shape
        zed_shape = episode_group["zed_rgb"].shape
        
        if len(rs_shape) == 4:  # (T, H, W, C)
            h, w, c = rs_shape[1], rs_shape[2], rs_shape[3]
            CONSOLE.log(f"RGB images: {h}x{w}x{c}")
            CONSOLE.log(f"  Current config: [2, 84, 84, 3]")
            CONSOLE.log(f"  Suggested:      [2, {h}, {w}, {c}]")
    
    # Check depth shapes
    if "rs_depth" in episode_group and "zed_depth" in episode_group:
        depth_shape = episode_group["rs_depth"].shape
        if len(depth_shape) == 3:  # (T, H, W)
            h, w = depth_shape[1], depth_shape[2]
            CONSOLE.log(f"Depth images: {h}x{w}")
            CONSOLE.log(f"  Current config: [2, 84, 84]")
            CONSOLE.log(f"  Suggested:      [2, {h}, {w}]")
    
    # Check point cloud shape
    if "zed_pcd" in episode_group:
        pcd_shape = episode_group["zed_pcd"].shape
        if len(pcd_shape) == 3:  # (T, N_points, features)
            n_points, features = pcd_shape[1], pcd_shape[2]
            CONSOLE.log(f"Point clouds: {n_points} points x {features} features")
            CONSOLE.log(f"  Current config: [1024, 6]")
            CONSOLE.log(f"  Suggested:      [{n_points}, {features}]")
    
    # Check agent_pos shape
    if "agent_pos" in episode_group:
        pos_shape = episode_group["agent_pos"].shape
        if len(pos_shape) == 2:  # (T, features)
            features = pos_shape[1]
            CONSOLE.log(f"Agent positions: {features} features")
            CONSOLE.log(f"  Current config: [7]")
            CONSOLE.log(f"  Suggested:      [{features}]")
    
    # Check action shape
    if "action" in episode_group:
        action_shape = episode_group["action"].shape
        if len(action_shape) == 2:  # (T, features)
            features = action_shape[1]
            CONSOLE.log(f"Actions: {features} features")
            CONSOLE.log(f"  Current config: [7]")
            CONSOLE.log(f"  Suggested:      [{features}]")
    
    # Print complete suggested config
    CONSOLE.log(f"\n[bold cyan]Complete Suggested shape_meta:")
    print_suggested_yaml_config(episode_group)

def print_suggested_yaml_config(episode_group) -> None:
    """Print a complete YAML config suggestion."""
    
    config_lines = ["shape_meta:"]
    
    # RGB config
    if "rs_rgb" in episode_group:
        rgb_shape = episode_group["rs_rgb"].shape
        if len(rgb_shape) == 4:
            h, w, c = rgb_shape[1], rgb_shape[2], rgb_shape[3]
            config_lines.extend([
                "  rgb:",
                f"    shape: [2, {h}, {w}, {c}]",
                "    dtype: uint8"
            ])
    
    # Depth config
    if "rs_depth" in episode_group:
        depth_shape = episode_group["rs_depth"].shape
        if len(depth_shape) == 3:
            h, w = depth_shape[1], depth_shape[2]
            config_lines.extend([
                "  depth:",
                f"    shape: [2, {h}, {w}]",
                "    dtype: float32"
            ])
    
    # Point cloud config
    if "zed_pcd" in episode_group:
        pcd_shape = episode_group["zed_pcd"].shape
        if len(pcd_shape) == 3:
            n_points, features = pcd_shape[1], pcd_shape[2]
            config_lines.extend([
                "  point_cloud:",
                f"    shape: [{n_points}, {features}]",
                "    dtype: float32"
            ])
    
    # Agent position config
    if "agent_pos" in episode_group:
        pos_shape = episode_group["agent_pos"].shape
        if len(pos_shape) == 2:
            features = pos_shape[1]
            config_lines.extend([
                "  agent_pos:",
                f"    shape: [{features}]       # [x y z qw qx qy qz]",
                "    dtype: float32"
            ])
    
    # Action config
    if "action" in episode_group:
        action_shape = episode_group["action"].shape
        if len(action_shape) == 2:
            features = action_shape[1]
            config_lines.extend([
                "  action:",
                f"    shape: [{features}]",
                "    dtype: float32"
            ])
    
    # Print the config
    for line in config_lines:
        CONSOLE.log(f"[cyan]{line}")

def main():
    """Main function to run the inspector."""
    
    # Path to your Zarr dataset
    ZARR_PATH = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline_zarr")
    
    # Inspect the first episode (you can change this)
    EPISODE_IDX = 0
    
    CONSOLE.log(f"[bold blue]Inspecting Zarr Dataset")
    CONSOLE.log(f"Path: {ZARR_PATH}")
    CONSOLE.log(f"Episode: {EPISODE_IDX}")
    
    inspect_zarr_dataset(ZARR_PATH, EPISODE_IDX)

if __name__ == "__main__":
    main()