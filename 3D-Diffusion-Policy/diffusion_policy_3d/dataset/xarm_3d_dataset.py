#!/usr/bin/env python3
"""
XArmPickMultiCamDataset
=======================

Loads the Zarr produced by `full_convert_zarr.py` and returns samples ready for DP3.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import zarr
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def _resize(img: np.ndarray, size: int = 84) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class XArmBaselineDataset(BaseDataset):
    def __init__(self,zarr_path: str, horizon: int = 16, future_action: int = 8, image_size: int = 84, n_points: int = 1024) -> None:
        # import pdb; pdb.set_trace()
        self.root = zarr.open(zarr_path, mode="r")
        self.keys: List[str] = sorted(self.root.group_keys())

        self.H = horizon
        self.F = future_action
        self.size = image_size
        self.n_points = n_points

        # build flat index → (episode_key, start_t)
        self.index: List[Tuple[str, int]] = []
        for k in self.keys:
            T = self.root[k].attrs["length"]
            self.index += [(k, t) for t in range(T - self.H - self.F)]

    # ────────────────────────────────
    def __len__(self) -> int:
        return len(self.index)

    # ────────────────────────────────
    def __getitem__(self, idx: int):
        epi, t0 = self.index[idx]
        grp = self.root[epi]

        # slice windows
        past_sl = slice(t0, t0 + self.H)
        fut_sl  = slice(t0, t0 + self.F)

        # Only want realsense rgb
        rgb  = grp["rs_rgb"][past_sl]         # (H, H_rs, W_rs, 3)

        # resize + pack cameras axis-1
        rgb   = np.stack([_resize(rgb[i]) for i in range(self.H)], axis=0)        # (H, 2, 84, 84, 3)

        # point cloud: FPS-downsample to n_points
        pcd_all = grp["zed_pcd"][past_sl]               # (H, Np, 6)
        if pcd_all.shape[1] > self.n_points:
            idxs = np.random.choice(pcd_all.shape[1], self.n_points, replace=False)
            pcd_all = pcd_all[:, idxs]
        else:
            # zero-pad if Np < n_points
            pad = self.n_points - pcd_all.shape[1]
            pcd_all = np.pad(pcd_all, ((0, 0), (0, pad), (0, 0)), mode="constant")

        obs_dict = dict(
            rgb        = rgb.astype(np.uint8),
            point_cloud= pcd_all.astype(np.float32),
            agent_pos  = grp["agent_pos"][past_sl].astype(np.float32),
        )

        sample = {
            "obs": obs_dict,
            "action": grp["action"][fut_sl].astype(np.float32),
        }
        return sample

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Return a LinearNormalizer for fields the policy uses.
        Only agent_pos is normalised; rgb/depth/pc remain raw
        (identity normaliser) just like in the authors’ DexArtDataset.
        """
        from diffusion_policy_3d.model.common.normalizer import (
            LinearNormalizer, SingleFieldLinearNormalizer)

        # --- stack across all timesteps of all episodes ---
        poses = []
        actions = []
        for epi in self.keys:
            poses.append(self.root[epi]["agent_pos"][...])
            actions.append(self.root[epi]["action"][...])
        poses   = np.concatenate(poses,   axis=0)   # (total_T, 7)
        actions = np.concatenate(actions, axis=0)   # (total_T, 7)

        data_for_stats = {
            "agent_pos": poses.astype(np.float32),
            "action":    actions.astype(np.float32),
        }

        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data_for_stats,
            last_n_dims=1,        # compute over the vector dim only
            mode=mode,
            **kwargs
        )

        # fields we leave untouched
        for k in ["rgb", "depth", "point_cloud"]:
            normalizer[k] = SingleFieldLinearNormalizer.create_identity()

        return normalizer
