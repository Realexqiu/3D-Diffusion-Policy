#!/usr/bin/env python3
"""
Integrated HDF5 → (GDINO + SAM-2) → Zarr converter
with automatic mask & frame dumping.

If DEBUGGING is True we process only the first episode (episode_0000);
otherwise we process every .hdf5 file.

For *each* episode we now create:
  extras_dir/
     episode_0000/
        masks/                 ← all per-frame mask PNGs
        rs_rgb_first.png
        rs_depth_first.png
        zed_rgb_first.png
        zed_depth_first.png
        rs_rgb_last.png
        rs_depth_last.png
        zed_rgb_last.png
        zed_depth_last.png
"""

# ────────────────────────────────────────────────────────────────
#  Standard imports
# ────────────────────────────────────────────────────────────────
import os, glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from PIL import Image
import h5py
import numpy as np
import torch
import zarr
from natsort import natsorted
from rich.console import Console
from tqdm import tqdm

from inference.custom_gsam_utils_v2 import GroundedSAM2 as GSamUtil, default_gsam_config

DEBUGGING = False  # if True: only process episode_0000
CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def hdf5_to_dict(pth: str) -> Dict[str, np.ndarray]:
    with h5py.File(pth, "r") as f:
        return {k: f[k][()] for k in f.keys()}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_depth_png(depth: np.ndarray, out_file: Path) -> None:
    """Normalize a float32 depth map, apply viridis colormap, save to PNG."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.max() == depth.min():
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out_file), color)

def dump_first_last_frames(rs_rgb: np.ndarray, rs_depth: np.ndarray, zed_rgb: np.ndarray, zed_depth: np.ndarray, ep_dir: Path) -> None:
    """Save first/last RGB & depth frames for RS and ZED cameras."""
    ensure_dir(ep_dir)
    first, last = 0, rs_rgb.shape[0] - 1
    for idx, tag in [(first, "first"), (last, "last")]:
        # RS RGB
        cv2.imwrite(str(ep_dir / f"rs_rgb_{tag}.png"), rs_rgb[idx])
        # RS depth
        save_depth_png(rs_depth[idx], ep_dir / f"rs_depth_{tag}.png")
        # ZED RGB
        cv2.imwrite(str(ep_dir / f"zed_rgb_{tag}.png"), zed_rgb[idx])
        # ZED depth
        save_depth_png(zed_depth[idx], ep_dir / f"zed_depth_{tag}.png")

def check_episode_complete(zarr_group, episode_name: str) -> bool:
    """Check if an episode has been completely processed."""
    if episode_name not in zarr_group:
        return False
    
    ep_group = zarr_group[episode_name]
    required_arrays = ["rs_rgb", "rs_depth", "zed_rgb", "zed_depth", "zed_pcd", "agent_pos", "action"]
    
    for array_name in required_arrays:
        if array_name not in ep_group:
            CONSOLE.log(f"[yellow]Episode {episode_name} missing {array_name}, will reprocess")
            return False
    
    # Check if the episode has the expected length attribute
    if "length" not in ep_group.attrs:
        CONSOLE.log(f"[yellow]Episode {episode_name} missing length attribute, will reprocess")
        return False
    
    return True

def get_processed_episodes(zarr_dir: Path) -> set:
    """Get set of already processed episode names."""
    processed = set()
    
    if not zarr_dir.exists():
        return processed
    
    try:
        root = zarr.open(zarr_dir, mode="r")
        for key in root.keys():
            if key.startswith("episode_") and check_episode_complete(root, key):
                processed.add(key)
        CONSOLE.log(f"[green]Found {len(processed)} already processed episodes")
        if processed:
            sorted_episodes = sorted(processed)
            CONSOLE.log(f"[green]Processed episodes: {sorted_episodes[0]} to {sorted_episodes[-1]}")
    except Exception as e:
        CONSOLE.log(f"[yellow]Could not read existing Zarr file: {e}")
    
    return processed

# ────────────────────────────────────────────────────────────────
#  GSAM wrapper
# ────────────────────────────────────────────────────────────────
class GSAM2:
    def __init__(self):
        self.cfg  = default_gsam_config()
        self.util = GSamUtil(cfg=self.cfg, use_weights=True)

    def process_episode(self, frames: np.ndarray, depths: np.ndarray, mask_dump_dir: Path) -> List[np.ndarray]:
        """Runs Grounded SAM-2 tracking + PCD generation for an episode."""
        T = len(frames)
        ensure_dir(mask_dump_dir)

        # ── detect boxes on the first frame ─────────────────────────
        boxes, ok = self.util.detect_boxes(frames[0])
        if not ok:
            empty = np.zeros((T, self.cfg["pts_per_pcd"], 6), np.float32)
            return [pc for pc in empty]

        # ── initialise masks & tracking ─────────────────────────────
        init_masks = self.util.init_track(frames[0], boxes)

        # decide storage based on weighted flag
        if self.util.use_weights:
            masks: Dict[int, Union[np.ndarray, Dict[int, np.ndarray]]] = {0: init_masks}
        else:
            combined0 = np.zeros_like(next(iter(init_masks.values())), np.uint8)
            for m in init_masks.values():
                combined0 |= m
            masks = {0: combined0}

        # dump initial masks
        for obj_id, m in init_masks.items():
            Image.fromarray(m).save(mask_dump_dir / f"frame_0000_obj{obj_id}_mask.png")
        if not self.util.use_weights:
            Image.fromarray(masks[0]).save(mask_dump_dir / "frame_0000_combined_mask.png")

        # ── propagate masks through subsequent frames ───────────────
        for t in range(1, T):
            prop = self.util.propagate(frames[t])
            if isinstance(prop, dict):
                if self.util.use_weights:
                    masks[t] = prop
                else:
                    combined = np.zeros_like(next(iter(prop.values())), np.uint8)
                    for m in prop.values():
                        combined |= m
                    masks[t] = combined
            else:
                masks[t] = prop

        # dump propagated masks
        for t, m in masks.items():
            if isinstance(m, np.ndarray):
                Image.fromarray(m).save(mask_dump_dir / f"mask_{t:04d}.png")

        # ── generate point clouds ───────────────────────────────────
        pcds = []
        for t in range(T):
            mask_input = masks[t]
            pcd = self.util._make_pcd(mask_input, depths[t], frames[t])
            pcds.append(pcd)
        return pcds

# ────────────────────────────────────────────────────────────────
#  Main conversion routine
# ────────────────────────────────────────────────────────────────
def main() -> None:
    H5_DIR  = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline")
    ZARR_DIR = Path("/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/new_setup_100_baseline_zarr")
    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    # extras directory lives next to the Zarr output
    if ZARR_DIR.name.endswith("_zarr"):
        debug_name = ZARR_DIR.name.replace("_zarr", "_debug")
    else:
        debug_name = f"{ZARR_DIR.name}_debug"
    DEBUG_DIR = ZARR_DIR.parent / debug_name
    ensure_dir(DEBUG_DIR)

    files = natsorted(glob.glob(str(H5_DIR / "*.hdf5")))
    if not files:
        CONSOLE.log(f"[red]No HDF5 files found in {H5_DIR}")
        return

    # Check which episodes are already processed
    processed_episodes = get_processed_episodes(ZARR_DIR)
    
    gsam = GSAM2()
    
    # Open Zarr in append mode to add new episodes
    root = zarr.open(ZARR_DIR, mode="a")

    # Count episodes to process
    episodes_to_process = []
    for idx, path in enumerate(files):
        episode_name = f"episode_{idx:04d}"
        if episode_name not in processed_episodes:
            episodes_to_process.append((idx, path, episode_name))
        if DEBUGGING and len(episodes_to_process) >= 1:
            break

    if not episodes_to_process:
        CONSOLE.log("[green]All episodes already processed!")
        return

    CONSOLE.log(f"[blue]Processing {len(episodes_to_process)} remaining episodes...")

    # ────────────────────────────────────────────────────────────
    for idx, path, episode_name in tqdm(episodes_to_process, desc="Episodes"):
        CONSOLE.log(f"[blue]Processing {episode_name} ({idx+1}/{len(files)})")

        # ―― load episode ―――――――――――――――――――――――――――――――――――――
        try:
            data = hdf5_to_dict(path)
        except Exception as e:
            CONSOLE.log(f"[red]Error loading {path}: {e}")
            continue

        T = data["pose"].shape[0]
        rs_rgb = data["rs_color_images"]
        rs_depth = data["rs_depth_images"]
        zed_rgb = data["zed_color_images"]
        zed_depth= data["zed_depth_images"]

        # episode-specific extras directory
        ep_dir = DEBUG_DIR / episode_name
        mask_dir = ep_dir / "masks"

        try:
            # ―― GSAM → masks → point clouds ―――――――――――――――――――――
            pcds = gsam.process_episode(zed_rgb, zed_depth, mask_dump_dir=mask_dir)
            pcds_arr = np.stack(pcds, axis=0)  # (T, K, 6)

            # ―― save first / last frames ――――――――――――――――――――――
            dump_first_last_frames(rs_rgb, rs_depth, zed_rgb, zed_depth, ep_dir)

            # ―― store episode to Zarr ―――――――――――――――――――――――――
            # Remove existing group if it exists (in case of partial processing)
            if episode_name in root:
                del root[episode_name]
            
            grp = root.create_group(episode_name)
            grp.array("rs_rgb",    rs_rgb,    dtype=np.uint8,  chunks=(1, *rs_rgb.shape[1:]))
            grp.array("rs_depth",  rs_depth,  dtype=np.float32,chunks=(1, *rs_depth.shape[1:]))
            grp.array("zed_rgb",   zed_rgb,   dtype=np.uint8,  chunks=(1, *zed_rgb.shape[1:]))
            grp.array("zed_depth", zed_depth, dtype=np.float32,chunks=(1, *zed_depth.shape[1:]))
            grp.array("zed_pcd",   pcds_arr,  dtype=np.float32,chunks=(1, *pcds_arr.shape[1:]))
            grp.array("agent_pos", data["pose"],               dtype=np.float32)
            grp.array("action",    data["last_pose"] - data["pose"], dtype=np.float32)
            grp.attrs["length"] = T

            CONSOLE.log(f"[green]✓ Completed {episode_name}")
            
            # Reset GSAM state for next episode to avoid memory issues
            gsam.util.reset()
            
        except Exception as e:
            CONSOLE.log(f"[red]Error processing {episode_name}: {e}")
            # Clean up partial episode data
            if episode_name in root:
                del root[episode_name]
            continue

    processed_count = len(episodes_to_process)
    total_count = 1 if DEBUGGING else len(files)
    CONSOLE.log(f"[green]Finished.[/] Processed {processed_count} new episodes")
    CONSOLE.log(f"[green]Total episodes in Zarr: {len(root.keys())}/{total_count}")
    CONSOLE.log(f"[green]Extra masks & frames saved under {DEBUG_DIR}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
