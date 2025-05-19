#!/usr/bin/env python3
"""
Fast HDF5 → mask-propagation → point-cloud converter
----------------------------------------------------
1. Saves all RGB frames from the HDF5 to disk.
2. Runs Grounding DINO **once** on frame 0 to get object boxes.
3. Seeds a SAM-2 *video* predictor with those boxes and
   propagates the masks through the whole sequence.
4. Writes binary masks, overlay visualisations, and coloured
   point-clouds (PLY) for every frame.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
import open3d as o3d
import torch
from rich.console import Console
from rich.progress import track
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2_video_predictor

from groundingdino.util.inference import load_model, predict, load_image

CONSOLE = Console()

def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Logical-OR combine a list of H×W boolean masks → uint8 {0,255}."""
    if not masks:
        return np.zeros(0, dtype=np.uint8)
    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        merged |= m.astype(bool)
    return (merged * 255).astype(np.uint8)

class HDF5GSAM2Processor:
    """
    One-shot GDINO + SAM-2 mask propagation → coloured point clouds.
    """

    # ── initialisation ────────────────────────────────────────────
    def __init__(self):
        # ← paths ---------------------------------------------------
        hdf5_path = "/home/alex/Documents/robot_learning/robot_learning/data_collection_ros2/rs_zed_dt/3d_strawbery_50/episode_0.hdf5"
        output_dir = "/home/alex/Documents/robot_learning/robot_learning/inference/gsam2_test_output"
        self.hdf5_path = Path(hdf5_path).expanduser()
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.masks_dir = self.output_dir / "masks"
        self.vis_dir = self.output_dir / "visualizations"
        self.pcd_dir = self.output_dir / "pointclouds"
        for d in (self.frames_dir, self.masks_dir, self.vis_dir, self.pcd_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ← camera intrinsics (ZED 2K) ------------------------------
        self.fx, self.fy = 1069.73, 1069.73
        self.cx, self.cy = 1135.86, 680.69

        # ← Grounding DINO settings --------------------------------
        self.gdino_ckpt = "/home/alex/Documents/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.gdino_cfg = "/home/alex/Documents/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.box_thresh = 0.3
        self.text_thresh = 0.2

        # ← SAM-2 settings -----------------------------------------
        self.sam2_ckpt = "/home/alex/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_base_plus.pt"
        self.sam2_cfg = "//home/alex/Documents/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"

        # ← device -------------------------------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        CONSOLE.log(f"[bold blue]Using device:[/] {self.device}")

        # ← internal state -----------------------------------------
        self.depth_cache: List[np.ndarray] = []  # depth per frame
        self.masks: Dict[int, np.ndarray] = {}   # frame_idx → mask
        self.num_frames: int = 0
        self.H = self.W = 0

        # load models
        self._init_models()
        # gather meta info + dump RGB frames
        self._first_pass_save_frames()

    # ── model init ────────────────────────────────────────────────
    def _init_models(self):
        CONSOLE.log("[bold blue]Initialising models…")
        # GDINO (once)
        self.gdino = load_model(
            model_config_path=self.gdino_cfg,
            model_checkpoint_path=self.gdino_ckpt,
            device=self.device,
        )
        # SAM-2 video predictor
        self.video_predictor = build_sam2_video_predictor(
            self.sam2_cfg, self.sam2_ckpt, device=self.device
        )
        CONSOLE.log("[bold green]Models ready.")

    # ── first pass: save RGB + cache depth ────────────────────────
    def _first_pass_save_frames(self):
        CONSOLE.log("[bold blue]Pass 1/2 – dumping frames…")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)

        with h5py.File(self.hdf5_path, "r") as f:
            rgb_ds = f["zed_color_images"]
            depth_ds = f["zed_depth_images"]
            self.num_frames = rgb_ds.shape[0]
            self.H, self.W = rgb_ds.shape[1:3]

            for i in track(range(self.num_frames), description="Saving RGB"):
                rgb = rgb_ds[i]  # already BGR
                cv2.imwrite(str(self.frames_dir / f"{i:04d}.jpg"), rgb)
                self.depth_cache.append(depth_ds[i].astype(np.float32))
        CONSOLE.log("[bold green]Frames saved.")

    # ── GDINO on frame 0 → boxes ──────────────────────────────────
    def _run_gdino_once(self) -> np.ndarray:
        frame0_path = str(self.frames_dir / "0000.jpg")
        image_src, proc_img = load_image(frame0_path)

        # First query for strawberry
        boxes_strawberry, logits_strawberry, _ = predict(
            model=self.gdino,
            image=proc_img,
            caption="red strawberry",
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
        )
        
        # Second query for robot
        boxes_robot, logits_robot, _ = predict(
            model=self.gdino,
            image=proc_img,
            caption="robot",
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
        )
        
        # Combine boxes
        boxes = torch.cat([boxes_strawberry, boxes_robot], dim=0)
        if boxes.numel() == 0:
            CONSOLE.log("[bold red]Grounding DINO found no boxes. Exiting.")
            raise SystemExit

        # scale to pixel coords & xyxy
        boxes = boxes * torch.tensor([self.W, self.H, self.W, self.H], device=boxes.device)
        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        CONSOLE.log(f"[bold green]GDINO detected {len(boxes_xyxy)} box(es).")
        return boxes_xyxy

    # ── mask propagation via SAM-2 ────────────────────────────────
    def _propagate_masks(self):
        CONSOLE.log("[bold blue]Pass 2/2 – SAM-2 propagation…")

        # init state on the dumped frame directory
        state = self.video_predictor.init_state(video_path=str(self.frames_dir))

        # seed objects using GDINO boxes on frame 0
        for obj_id, box in enumerate(self._run_gdino_once(), start=1):
            # float32 as required
            box = box.astype(np.float32)
            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )

        # propagate through all frames
        for f_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(state):
            masks_this_frame = [
                (mask_logits[i] > 0).cpu().numpy()
                for i in range(len(obj_ids))
            ]
            self.masks[f_idx] = combine_masks(masks_this_frame)

        CONSOLE.log("[bold green]Masks propagated.")

    # ── depth+mask → coloured point cloud ─────────────────────────
    def _mask_depth_to_pcd(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        color_bgr: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        # Return empty point cloud if mask is empty
        if mask.sum() == 0:
            return o3d.geometry.PointCloud()

        # Find all non-zero pixels in the mask (these are our object pixels)
        v_idx, u_idx = np.where(mask > 0)
        
        # Get depth values for these pixels
        Z = depth[v_idx, u_idx]
        
        # Filter out invalid depth measurements (zeros or NaNs)
        valid = np.isfinite(Z) & (Z > 1e-6)
        if not np.any(valid):
            return o3d.geometry.PointCloud()

        # Get the valid pixel coordinates and depths
        u, v, Z = u_idx[valid], v_idx[valid], Z[valid]
        
        # Convert from pixel coordinates to 3D world coordinates using camera intrinsics
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # Stack X, Y, Z coordinates into a single Nx3 array of 3D points
        pcd_numpy = np.column_stack((X, Y, Z))
        
        # Get the colors for each point (convert BGR to RGB and normalize to [0,1])
        cols = color_bgr[v, u][:, ::-1] / 255.0

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
        return pcd
    
    # ── visualize a single point cloud ──────────────────────────────
    def _visualize_point_cloud(self, pcd, title):
        """
        Visualize a single point cloud in its own window
        """
        if pcd is None or len(pcd.points) == 0:
            CONSOLE.log(f"[bold red]Cannot visualize {title} - point cloud is empty")
            return
            
        # Create visualization window
        CONSOLE.log(f"[bold blue]Visualizing {title}...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Add geometries to visualizer
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # Set camera position for a good view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Set rendering options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        render_option.point_size = 2.0
        
        # Run the visualizer
        CONSOLE.log(f"[bold green]{title} ready. Close the window to continue.")
        vis.run()
        vis.destroy_window()

    # ── main entry ────────────────────────────────────────────────
    def run(self):
        self._propagate_masks()  # fills self.masks

        # iterate frames again to save masks, vis, and PLY
        for i in track(range(self.num_frames), description="Saving outputs"):
            rgb_path = self.frames_dir / f"{i:04d}.jpg"
            rgb = cv2.imread(str(rgb_path))  # BGR
            mask = self.masks.get(i, np.zeros((self.H, self.W), dtype=np.uint8))

            # Debug: check if RGB is loaded correctly
            if rgb is None or rgb.size == 0:
                CONSOLE.log(f"[bold red]Error loading RGB image {rgb_path}")
                continue

            # Debug: print shapes before resizing
            CONSOLE.log(f"[bold blue]Frame {i}: RGB shape: {rgb.shape}, Mask shape: {mask.shape}")

            # Handle 3D mask - squeeze out the extra dimension if needed
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                CONSOLE.log(f"[bold yellow]Squeezing 3D mask for frame {i}")
                mask = np.squeeze(mask, axis=0)

            # Check if mask is empty or has improper dimensions
            if mask.size == 0:
                CONSOLE.log(f"[bold red]Empty mask for frame {i}")
                mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            elif len(mask.shape) == 1:
                CONSOLE.log(f"[bold red]1D mask detected for frame {i}, reshaping")
                # Try to reshape 1D mask to expected dimensions
                mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            elif mask.shape[:2] != rgb.shape[:2]:
                CONSOLE.log(f"[bold red]Mask size mismatch: {mask.shape[:2]} vs RGB: {rgb.shape[:2]}")
                # Resize only if mask has valid dimensions
                if mask.shape[0] > 0 and mask.shape[1] > 0 and rgb.shape[0] > 0 and rgb.shape[1] > 0:
                    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

            # write binary mask
            cv2.imwrite(str(self.masks_dir / f"{i:04d}.jpg"), mask)            

            # overlay visualisation (green)
            overlay = rgb.copy()
            overlay[mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
            cv2.imwrite(str(self.vis_dir / f"{i:04d}.jpg"), vis)

            # point cloud
            pcd = self._mask_depth_to_pcd(mask, self.depth_cache[i], rgb)
            if len(pcd.points) > 0:
                o3d.io.write_point_cloud(
                    str(self.pcd_dir / f"{i:04d}.ply"), pcd, write_ascii=False
                )

        CONSOLE.log("[bold green]All done – fast pipeline complete!")

        # Visualize first and last point clouds in separate windows
        first_pcd_path = self.pcd_dir / "0000.ply"
        last_pcd_path = self.pcd_dir / f"{self.num_frames-1:04d}.ply"
        
        # Load the point clouds from saved PLY files
        first_pcd = o3d.io.read_point_cloud(str(first_pcd_path))
        last_pcd = o3d.io.read_point_cloud(str(last_pcd_path))
        
        # Visualize in separate windows
        self._visualize_point_cloud(first_pcd, "First Frame Point Cloud")
        self._visualize_point_cloud(last_pcd, "Last Frame Point Cloud")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processor = HDF5GSAM2Processor()
    processor.run()
    
    # Print locations of saved files
    print(f"\nOutput files saved to:")
    print(f"  - RGB frames: {processor.frames_dir}")
    print(f"  - Binary masks: {processor.masks_dir}")
    print(f"  - Visualizations: {processor.vis_dir}")
    print(f"  - Point clouds: {processor.pcd_dir}")