# diffusion_policy/dataset/xarm_2d_dataset.py
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


class XArmImageDataset2D(BaseImageDataset):
    """
    Dataset for *2-D* Diffusion Policy on your xArm recordings.
    Each observation = RealSense RGB frame + 7-D agent_pos.

    Bypasses ReplayBuffer and reads directly from a filtered Zarr.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        # Open the Zarr store
        self.root = zarr.open(zarr_path, mode="r")
        # List of episodes
        self.episodes: List[str] = sorted(self.root.group_keys())
        # Split episodes
        np.random.seed(seed)
        n_eps = len(self.episodes)
        idxs = np.arange(n_eps)
        np.random.shuffle(idxs)
        n_val = int(val_ratio * n_eps)
        val_idxs = set(idxs[:n_val])
        train_idxs = [i for i in idxs if i not in val_idxs]
        if max_train_episodes is not None:
            train_idxs = list(train_idxs)[:max_train_episodes]
        self.train_eps = {self.episodes[i] for i in train_idxs}
        self.val_eps = {self.episodes[i] for i in val_idxs}
        # Build index list: (episode, start_t) for training
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.index: List[Tuple[str,int]] = []
        for epi in self.train_eps:
            grp = self.root[epi]
            T = grp["rs_rgb"].shape[0]
            for t in range(T - horizon):
                self.index.append((epi, t))

    def get_validation_dataset(self) -> "XArmImageDataset2D":
        # shallow copy, swap to validation episodes
        val_ds = copy.copy(self)
        val_ds.index = []
        for epi in self.val_eps:
            grp = val_ds.root[epi]
            T = grp["rs_rgb"].shape[0]
            for t in range(T - val_ds.horizon):
                val_ds.index.append((epi, t))
        return val_ds

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, t0 = self.index[idx]
        grp = self.root[epi]
        sl = slice(t0, t0 + self.horizon)
        # Load RGB frames
        rgb = grp["rs_rgb"][sl]        # (H, h, w, 3)
        # Agent positions and actions
        agent_pos = grp["agent_pos"][sl]  # (H, 7)
        action = grp["action"][sl]        # (H, 7)
        # Preprocess image: uint8 -> float32 [0,1], channel-first
        img = np.moveaxis(rgb, -1, 1).astype(np.float32) / 255.0
        sample = {
            "obs": {
                "image": img,
                "agent_pos": agent_pos.astype(np.float32),
            },
            "action": action.astype(np.float32),
        }
        return dict_apply(sample, torch.from_numpy)

    def get_normalizer(self, mode: str = "limits", **kwargs):
        # Gather all agent_pos and actions from training episodes
        poses = []
        acts = []
        for epi in self.train_eps:
            grp = self.root[epi]
            poses.append(grp["agent_pos"][...])
            acts.append(grp["action"][...])
        poses = np.concatenate(poses, axis=0)
        acts = np.concatenate(acts, axis=0)
        data = {"agent_pos": poses, "action": acts}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # Identity for image range
        normalizer["image"] = get_image_range_normalizer()
        return normalizer
