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
    Each observation = RealSense RGB frame + 8-D pose.
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 2,
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
            T = grp["rs_side_rgb"].shape[0]
            T_zed = grp["zed_rgb"].shape[0]
            for t in range(min(T, T_zed) - horizon):
                self.index.append((epi, t))

    def get_validation_dataset(self) -> "XArmImageDataset2D":
        # shallow copy, swap to validation episodes
        val_ds = copy.copy(self)
        val_ds.index = []
        for epi in self.val_eps:
            grp = val_ds.root[epi]
            T = grp["rs_side_rgb"].shape[0]
            T_zed = grp["zed_rgb"].shape[0]
            for t in range(min(T, T_zed) - val_ds.horizon):
                val_ds.index.append((epi, t))
        return val_ds

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # import pdb; pdb.set_trace()
        epi, t0 = self.index[idx]
        grp = self.root[epi]
        sl = slice(t0, t0 + self.horizon)

        # Load RGB frames
        rs_side_rgb = grp["rs_side_rgb"][sl]        # (H, h, w, 3)
        zed_rgb = grp["zed_rgb"][sl]        # (H, h, w, 3)

        # Agent positions and actions
        pose = grp["pose"][sl]  # (H, 10)
        action = grp["action"][sl]        # (H, 10)

        # If your Zarr stores HWC, move channels to front; if it already stores CHW, do nothing.
        if rs_side_rgb.ndim == 4 and rs_side_rgb.shape[-1] == 3:
            # (T, H, W, C) -> (T, C, H, W)
            rs_side_img = np.moveaxis(rs_side_rgb, -1, 1)
            zed_img = np.moveaxis(zed_rgb, -1, 1)
        else:
            # assume already (T, C, H, W)
            rs_side_img = rs_side_rgb
            zed_img = zed_rgb

        rs_side_img = rs_side_img.astype(np.float32) / 255.0
        zed_img = zed_img.astype(np.float32) / 255.0

        # Debug visualization - only show randomly every ~1000 samples
        if np.random.randint(0, 1000) == 0:
            # print pose and action
            print(pose[0])
            # pose is xyz then 6D rotation

            print(action[0])

            # print shape of pose and action
            print(pose.shape)
            print(action.shape)

            import pdb; pdb.set_trace()
            import matplotlib.pyplot as plt
            plt.imshow(np.transpose(rs_side_img[0], (1, 2, 0)))
            plt.show()

            # show zed_img
            plt.imshow(np.transpose(zed_img[0], (1, 2, 0)))
            plt.show()

            # let's save these images and pose and action
            import cv2
            import os

            # Create a directory for saving images
            save_dir = "debug_images"
            os.makedirs(save_dir, exist_ok=True)
            
            # save rs_side_img
            # resize image to save with cv2
            # print image shape
            print(rs_side_img[0].shape)
            # shape is (3, 180, 220)
            # reshape to (180, 220, 3)
            rs_side_img_resized = np.transpose(rs_side_img[0], (1, 2, 0))
            cv2.imwrite(os.path.join(save_dir, "rs_side_img.png"), rs_side_img_resized)

            # save zed_img
            # resize image to save with cv2
            # print image shape
            print(zed_img[0].shape)
            # shape is (3, 180, 220)
            # reshape to (180, 220, 3)
            zed_img_resized = np.transpose(zed_img[0], (1, 2, 0))
            cv2.imwrite(os.path.join(save_dir, "zed_img.png"), zed_img_resized)
            
            # save pose
            np.save(os.path.join(save_dir, "pose.npy"), pose[0])

            # save action
            np.save(os.path.join(save_dir, "action.npy"), action[0])

 


        sample = {
            "obs": {
                "rs_side_rgb": rs_side_img,
                "zed_rgb": zed_img,
                "pose": pose.astype(np.float32),
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
            poses.append(grp["pose"][...])
            acts.append(grp["action"][...])
        poses = np.concatenate(poses, axis=0)
        acts = np.concatenate(acts, axis=0)
        data = {"pose": poses, "action": acts}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # Identity for image range
        normalizer["rs_side_rgb"] = get_image_range_normalizer()
        # normalizer["rs_wrist_rgb"] = get_image_range_normalizer()
        normalizer["zed_rgb"] = get_image_range_normalizer()
        return normalizer
