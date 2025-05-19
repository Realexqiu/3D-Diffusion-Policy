#!/usr/bin/env python3
"""
Integrated HDF5 -> (GDINO + SAM-2) -> Zarr converter with DEBUG mode
===================================================================

This script processes either the entire dataset or, if DEBUGGING=True, only
one episode (episode_0000). When debugging, every SAM-2 mask is dumped as a
PNG so you can visually verify segmentation.
"""

import os, glob
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
import zarr
from natsort import natsorted
from rich.console import Console
from torchvision.ops import box_convert
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model, predict, load_image

# ────────────────────────────────────────────────────────────────
#  Debug toggle
# ────────────────────────────────────────────────────────────────
DEBUGGING = False  # if True: only process episode_0000 and dump masks

# ────────────────────────────────────────────────────────────────
#  Configuration block
# ────────────────────────────────────────────────────────────────
CONFIG = {
    # Data I/O ----------------------------------------------------
    "HDF5_DIR": "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/3d_strawberry_baseline_50_hdf5",
    "OUT_PATH": "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/3d_strawberry_baseline/3d_strawberry_baseline_50_zarr",

    # ZED image cropping
    "ZED_CROP": [400, 100, 1400, 2000],

    # ZED left intrinsics (after crop)
    "zed_fx": 1069.73,
    "zed_fy": 1069.73,
    "zed_cx": 1135.86 - 400,
    "zed_cy": 680.69 - 100,

    # Hard-coded ZED→xArm base transform
    "T_base_zed": np.array([
        [0.0200602,  0.7492477, -0.6619859, 0.580],
        [0.9983952,  0.0200602,  0.0529589, 0.020],
        [0.0529589, -0.6619859, -0.7476429, 0.570],
        [0.000,      0.000,      0.000,      1.000],
    ], dtype=np.float32),

    # Grounding-DINO ---------------------------------------------
    "GDINO_CKPT": "/home/alex/Documents/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
    "GDINO_CFG":  "/home/alex/Documents/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "BOX_THRESH": 0.30,
    "TEXT_THRESH": 0.20,
    "GDINO_QUERIES": ["red strawberry", "robot"],

    # SAM-2 -------------------------------------------------------
    "SAM2_CKPT": "/home/alex/Documents/segment-anything-2-real-time/checkpoints/sam2.1_hiera_base_plus.pt",
    "SAM2_CFG":  "//home/alex/Documents/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",

    # Point cloud settings ----------------------------------------
    "POINTS_PER_CLOUD": 4096,  # ensures fixed-size output
}
CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0,0), dtype=np.uint8)
    m = np.zeros_like(masks[0], dtype=bool)
    for x in masks:
        m |= x.astype(bool)
    return (m * 255).astype(np.uint8)

def downsample_point_cloud(pc: np.ndarray, target: int) -> np.ndarray:
    n = pc.shape[0]
    if n >= target:
        idx = np.random.choice(n, target, replace=False)
        return pc[idx]
    out = np.zeros((target, 6), dtype=np.float32)
    out[:n] = pc
    return out

def depth_to_point_cloud(mask: np.ndarray, depth: np.ndarray, rgb_bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float, T: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    v, u = np.where(mask > 0)
    Z = depth[v, u]
    valid = np.isfinite(Z) & (Z > 1e-6)
    if not valid.any():
        return np.empty((0,6), dtype=np.float32)
    u, v, Z = u[valid], v[valid], Z[valid]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
    pts_base = (T @ pts_cam.T).T[:, :3]
    rgb = rgb_bgr[v, u][:, ::-1] / 255.0
    return np.hstack([pts_base, rgb]).astype(np.float32)


def hdf5_to_dict(pth: str) -> Dict[str, np.ndarray]:
    with h5py.File(pth, 'r') as f:
        return {k: f[k][()] for k in f.keys()}

# ────────────────────────────────────────────────────────────────
#  GDINO + SAM-2 pipeline with mask-dump logging
# ────────────────────────────────────────────────────────────────
class GSAM2:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        CONSOLE.log(f"[blue]Init models on {self.device}…")
        self.gdino = load_model(model_config_path=cfg["GDINO_CFG"], model_checkpoint_path=cfg["GDINO_CKPT"], device=self.device)
        self.video_predictor = build_sam2_video_predictor(cfg["SAM2_CFG"], cfg["SAM2_CKPT"], device=self.device)
        CONSOLE.log("[green]Models ready.")

    def process_episode(self, zed_rgb_seq: np.ndarray, zed_depth_seq: np.ndarray, mask_dump_dir: Optional[Path] = None) -> List[np.ndarray]:
        T, H, W, _ = zed_rgb_seq.shape
        masks: Dict[int, np.ndarray] = {}

        # Create a temporary directory to store video frames as JPEG images.
        tmp = Path("__tmp_gsam2_frames")
        tmp.mkdir(exist_ok=True)

        # Save each frame from the sequence.
        for t in range(T):
            cv2.imwrite(str(tmp / f"{t:04d}.jpg"), zed_rgb_seq[t])

        # Process the first frame to detect objects using GDINO.
        image, image_transformed = load_image(str(tmp / "0000.jpg"))
        boxes = []
        for caption in self.cfg["GDINO_QUERIES"]:
            boxes, logits_, phrases = predict(
                model=self.gdino,
                image=image_transformed,
                caption=caption,
                box_threshold=self.cfg["BOX_THRESH"],
                text_threshold=self.cfg["TEXT_THRESH"]
            )
            if boxes.numel():
                boxes.append(boxes)

        # If no boxes are detected, log the issue and return empty point clouds.
        if not boxes:
            CONSOLE.log("[red]No boxes found; skipping masks.")
            empty = np.zeros((self.cfg["POINTS_PER_CLOUD"], 6), dtype=np.float32)
            return [empty.copy() for _ in range(T)]

        # Concatenate boxes, rescale them to image dimensions, and convert to xyxy format.
        B = torch.cat(boxes, 0) * torch.tensor([W, H, W, H], device=boxes[0].device)
        xyxy = box_convert(B, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        # Initialize the SAM-2 video predictor with the temporary video.
        state = self.video_predictor.init_state(video_path=str(tmp))
        for i, box in enumerate(xyxy, start=1):
            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=i,
                box=box.astype(np.float32)
            )

        # Propagate predictions through the video and combine the masks for each frame.
        for f_idx, obj_ids, logits in self.video_predictor.propagate_in_video(state):
            mlist = [(logits[i] > 0).cpu().numpy() for i in range(len(obj_ids))]
            masks[f_idx] = combine_masks(mlist)

        # Optionally dump mask images for debugging.
        if mask_dump_dir:
            CONSOLE.log(f"[yellow]Dumping {len(masks)} masks to {mask_dump_dir}")
            mask_dump_dir.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image
            except ImportError:
                CONSOLE.log("[red]PIL not installed; cannot dump mask images.")
            else:
                for f in range(T):
                    m = masks.get(f, np.zeros((H, W), dtype=np.uint8))
                    # Ensure the mask is 2D.
                    if m.ndim != 2:
                        m = np.squeeze(m)
                    img = Image.fromarray(m.astype(np.uint8), mode='L')
                    outp = mask_dump_dir / f"mask_{f:04d}.png"
                    try:
                        img.save(str(outp))
                    except Exception as e:
                        CONSOLE.log(f"[red]Failed to save mask {f:04d}: {e}")

        # Build point clouds from the depth maps using the computed masks.
        clouds = []
        for t in range(T):
            m = masks.get(t, np.zeros((H, W), dtype=np.uint8))
            pc = depth_to_point_cloud(
                mask=m,
                depth=zed_depth_seq[t],
                rgb_bgr=zed_rgb_seq[t],
                fx=self.cfg["zed_fx"],
                fy=self.cfg["zed_fy"],
                cx=self.cfg["zed_cx"],
                cy=self.cfg["zed_cy"],
                T=self.cfg["T_base_zed"]
            )
            clouds.append(downsample_point_cloud(pc, self.cfg["POINTS_PER_CLOUD"]))

        # Clean up: remove temporary images and delete the temporary directory.
        for f in tmp.glob("*.jpg"):
            f.unlink()
        tmp.rmdir()

        return clouds

# ────────────────────────────────────────────────────────────────
#  Main conversion
# ────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    H5 = Path(cfg["HDF5_DIR"])
    OUT = Path(cfg["OUT_PATH"])
    OUT.mkdir(parents=True, exist_ok=True)

    files = natsorted(glob.glob(str(H5/"*.hdf5")))
    if not files:
        CONSOLE.log(f"[red]No HDF5 files in {H5}"); return

    g2 = GSAM2(cfg)
    root = zarr.open(OUT, mode="w")

    for idx, path in enumerate(tqdm(files, desc="Episodes")):
        if DEBUGGING and idx>0: break

        # load
        d = hdf5_to_dict(path)
        T = d["pose"].shape[0]
        rs_rgb, rs_depth = d["rs_color_images"], d["rs_depth_images"]
        zed_rgb, zed_depth = d["zed_color_images"], d["zed_depth_images"]
        x,y,w,h = cfg["ZED_CROP"]
        zed_rgb = zed_rgb[:,y:y+h,x:x+w]
        zed_depth = zed_depth[:,y:y+h,x:x+w]

        # prepare mask dump dir
        debug_dir = (OUT/"debug_masks"/f"episode_{idx:04d}") if DEBUGGING else None

        # process
        pcds = g2.process_episode(zed_rgb, zed_depth, mask_dump_dir=debug_dir)
        arr = np.stack(pcds,axis=0)

        # store
        grp = root.create_group(f"episode_{idx:04d}")
        grp.array("rs_rgb", rs_rgb, dtype=np.uint8, chunks=(1,*rs_rgb.shape[1:]))
        grp.array("rs_depth", rs_depth, dtype=np.float32, chunks=(1,*rs_depth.shape[1:]))
        grp.array("zed_rgb", zed_rgb, dtype=np.uint8, chunks=(1,*zed_rgb.shape[1:]))
        grp.array("zed_depth",zed_depth,dtype=np.float32,chunks=(1,*zed_depth.shape[1:]))
        grp.array("zed_pcd",arr,dtype=np.float32,chunks=(1,*arr.shape[1:]))
        grp.array("agent_pos",d["pose"],dtype=np.float32)
        grp.array("action",d["last_pose"]-d["pose"],dtype=np.float32)
        grp.attrs["length"] = T

    CONSOLE.log(f"[green]Wrote { (1 if DEBUGGING else len(files)) } episode(s) to {OUT }")

if __name__=="__main__": main()
