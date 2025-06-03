import tkinter as tk
from PIL import Image, ImageTk


import glob
import os
import shutil
from typing import Dict, Optional
from pathlib import Path
from gendp.common.part_level_semantics import ObjectSemantics
import torch
import numpy as np
import copy
import multiprocessing
import zarr
import time
from tqdm import tqdm


import torchvision.transforms as transforms

import concurrent.futures
import h5py
import cv2
import open3d as o3d
from filelock import FileLock
import transforms3d
import scipy.spatial.transform as st

import cv2
from PIL import Image, ImageDraw, ImageFont

import time


# import image
from PIL import Image

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gendp.common.kinematics_utils import KinHelper
from gendp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gendp.common.rob_mesh_utils import load_mesh, mesh_poses_to_pc
from gendp.common.data_utils import d3fields_proc, _convert_actions, load_dict_from_hdf5, modify_hdf5_from_dict, plot_3d_trajectory_with_gradient
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from gendp.common.normalize_util import (
    get_image_identity_range_normalizer,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

from gendp.dataset.utils import crop_points_within_bounds, downsample_with_fps, get_current_YYYY_MM_DD_hh_mm_ss_ms, compute_point_cloud, get_gsam2_masks, visualize_point_cloud, compute_densetact_image_difference

from ikpy.chain import Chain

import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt


# from torch_geometric.nn import fps
import pytorch3d.ops as torch3d_ops
import torchvision.transforms as T
from torchvision.ops import box_convert
import random

# add to sys path so grounding dino will show up
import sys

# TODO: Uncomment this eventually
# from sam2.build_sam import build_sam2_camera_predictor

sys.path.append("/arm/u/maestro/Desktop/DT-Diffusion-Policy/gendp/Grounded-SAM-2")
sys.path.append("/home/user/Documents/DT-Diffusion-Policy/gendp/Grounded-SAM-2")
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import grounding_dino.groundingdino.datasets.transforms as Tr
from sklearn.decomposition import PCA

import pinocchio as pin
import osqp
from scipy import sparse

from gendp.dataset.dinov2_featurizer import DINOV2Featurizer
from gendp.dataset.radio_featurizer import RADIOFeaturizer
from gendp.common.leaphandik import LeapHandIK
import gendp.constants as constants

register_codecs()


############## for ptcloud 
# Import for DenseTact pointcloud generation
sys.path.append('/home/user/Documents/DT-Diffusion-Policy/gendp/dt/scripts')
try:
    from pcl_inference import DenseTactPointCloudGenerator
    print("DenseTactPointCloudGenerator imported successfully.")
except ImportError:
    print("Warning: Failed to import DenseTactPointCloudGenerator. Make sure pcl_inference.py is in the correct path.")


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

###################

class DenseTactModel:
    def __init__(self):
        pass

    def __call__(self, dt_images):
        outputs = []
        for dt_image in dt_images:
            # output = [dt_image, dt_image]
            output = [dt_image]

            output = np.concatenate(output)
            outputs.append(output)

        return outputs
    
        

# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
# this is the MAIN function!
def _convert_real_to_dp_replay(store, shape_meta, dataset_dir, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, fusion=None, robot_name='panda', expected_labels=None,
        exclude_colors=[]):
    

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5
    densetact_model = DenseTactModel()
    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    spatial_keys = list()

    touch_rgb_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        if type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'spatial':
            spatial_keys.append(key)
            max_pts_num = obs_shape_meta[key]['shape'][1]
        elif type == 'touch_rgb':
            touch_rgb_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    episodes_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split('_')[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    spatial_data_dict = dict()
    touch_rgb_data_dict = dict()

    # leap = LeapHandIK()
    tf = RotationTransformer('rotation_6d', 'matrix')

    # TODO: Uncomment these out eventually
    # dinov2_model = DINOV2Featurizer()

    # construct the better model featurizer
    radio_model = RADIOFeaturizer(model_version="radio_v2.5-l")

    # features_path = '/home/user/Documents/DT-Diffusion-Policy/gendp/gendp/gendp/common/robot.npy'
    # object_semantics = ObjectSemantics(features_path)

    # sam2_predictor = build_sam2_camera_predictor(constants.SAM_MODEL_CFG, constants.SAM2_CHECKPOINT)
    # print("SAM Camera Predictor initialized")
    # grounding_model = load_model(
    #     model_config_path=constants.GROUNDING_DINO_CONFIG, 
    #     model_checkpoint_path=constants.GROUNDING_DINO_CHECKPOINT,
    #     device=constants.DEVICE
    # )
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    only_pose = False
    if_init = False 
    if 'info' in shape_meta['action'] and 'only_pose' in shape_meta['action']['info']:
        only_pose = shape_meta['action']['info']['only_pose']

    is_fruit = True
    if 'info' in shape_meta['action'] and 'is_fruit' in shape_meta['action']['info']:
        is_fruit = shape_meta['action']['info']['is_fruit']

    is_jparse = False
    if 'info' in shape_meta['action'] and 'is_jparse' in shape_meta['action']['info']:
        is_jparse = shape_meta['action']['info']['is_jparse']


    ################################## ptcloud- related

    # Initialize DenseTact pointcloud generators for each finger's sensor
    base_dir = '/home/user/Documents/DT-Diffusion-Policy/gendp/dt/scripts'
    model_name = 'hiera'  # Use the same model as in pcl_dt_publisher.py

    # pcl_generators = {
    #     'thumb': DenseTactPointCloudGenerator(
    #         base_dir=base_dir,
    #         sensor_name='es1t',  # Update with correct sensor names
    #         model_name=model_name,
    #         finger_name='thumb',
    #         downsample_factor=10,
    #         is_publish='none'  # Don't publish in ROS
    #     ),
    #     'pointer': DenseTactPointCloudGenerator(
    #         base_dir=base_dir,
    #         sensor_name='es3t',
    #         model_name=model_name,
    #         finger_name='pointer',
    #         downsample_factor=10,
    #         is_publish='none'
    #     ),
    #     'middle': DenseTactPointCloudGenerator(
    #         base_dir=base_dir,
    #         sensor_name='sf2t',
    #         model_name=model_name,
    #         finger_name='middle',
    #         downsample_factor=10,
    #         is_publish='none'
    #     )
    #     # Add more fingers if needed
    # }

    # # Provide LeapHandIK instance to the generators
    # for finger, generator in pcl_generators.items():
    #     generator.set_leap_hand_ik(leap)


    ################################## ptcloud- related
    
    # TODO: augment the point cloud data with color jittering
    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        # reset if_init
        if_init = False 
        dataset_path = os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5')
        print(f"Processing episode {dataset_path}")
        feats_per_epi = list() # save it separately to avoid OOM
        with h5py.File(dataset_path) as file:
            print(file)
            # count total steps
            if 'cartesian_action' in file:
                episode_length = file['cartesian_action'].shape[0]
            else:
                episode_length = file['pose'].shape[0]

            remove_last_step = True
            if remove_last_step:
                episode_length -= 1

            # we can add some simple extra augmentations with color jittering
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            positions_episode = []

            for key in lowdim_keys + ['action']:
                data_key = key
                if key == 'action':
                    data_key = 'action_cart_ang' if 'key' not in shape_meta['action'] else shape_meta['action']['key']
                elif key == 'pose':
                    data_key = 'pose'
                else:
                    data_key = 'prev_agent_pos_ang'
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()

                # get data, which is a. robot pose or b. action
                this_data = file[data_key][()] # type: ignore

                if is_fruit:
                    pass
                    # items 4 to 7 are xyzw format, swap to wxyz
                    # Extract the quaternion (xyzw)
                    # quat_xyzw = this_data[:, 3:8].copy()
                    
                    # # Rearrange from xyzw to wxyz format
                    # # Take the w component (last) and move it to the front
                    # this_data[:, 3] = quat_xyzw[:, 3]
                    # this_data[:, 4] = quat_xyzw[:, 0]
                    # this_data[:, 5] = quat_xyzw[:, 1]
                    # this_data[:, 6] = quat_xyzw[:, 2]

                # if gripper action is available, add it to the actions
                if 'gripper' in file:
                    gripper = file['gripper'][()]
                    gripper = gripper[..., None]

                    gripper = np.clip(gripper, -1, 1)

                    this_data = np.concatenate([this_data, gripper], axis=1) # type: ignore

                this_data = _convert_actions(
                    raw_actions=this_data,
                    rotation_transformer=rotation_transformer,
                    action_key=data_key,
                )

                # if key is action, we will shift the data by 1
                if is_jparse:
                    if key == 'action':
                        this_data = np.roll(this_data, -1, axis=0)
                    this_data = this_data[:-1] # type: ignore


                print(f"Min: {np.min(this_data)}, Max: {np.max(this_data)}")
                if is_fruit:
                    # if key is pose, the actions are the next timestep. We will shift the data by 1
                    action_data = None
                    if only_pose and key == 'action':
                        # shift with numpy roll
                        action_data = this_data.copy() # type: ignore
                        action_data = np.roll(action_data, -1, axis=0)
                        # take off last element
                        action_data = action_data[:-1]
                        this_data = this_data[:-1] # type: ignore
                    elif key == 'action':
                        action_data = this_data.copy() # type: ignore
                        action_data = np.roll(action_data, -1, axis=0)
                        action_data = action_data[:-1]
                        this_data = this_data[:-1] # type: ignore
                    else:
                        action_data = this_data[:-1]
                        this_data = action_data # type: ignore
                    
                    lowdim_data_dict[key].append(action_data)
                else:
                    lowdim_data_dict[key].append(this_data)

                # TODO: fix this code below.
                if only_pose and action_data is not None:
                    if 'pose' not in lowdim_data_dict:
                        lowdim_data_dict['pose'] = list()
                    lowdim_data_dict['pose'].append(this_data)
            
            # 3d plot
            # get first 3 components of pose, column wise
            # pose_points = np.array(lowdim_data_dict['pose'][-1])[:,:3]
            # plot_3d_trajectory_with_gradient(pose_points)

            # action_points = np.array(lowdim_data_dict['action'][-1])[:,:3]
            # plot_3d_trajectory_with_gradient(action_points)

            color_jitter = transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.0
            )

            random_crop = transforms.RandomCrop((180, 180))

            # go through any rgb image keys
            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                imgs = file[key][()] # type: ignore

                if 'zed' in key or 'rs' in key:
                    imgs = [img.transpose(1, 2, 0) for img in imgs] # type: ignore
                    imgs = [img[:, :, :3] for img in imgs] # type: ignore

                print("Read images", key)
                shape = tuple(shape_meta['obs'][key]['shape'])

                if 'dt' in key:
                    # crop the dt by only taking x > 50 and < 300
                    imgs = [img.transpose(1, 2, 0) for img in imgs] # type: ignore
                    imgs = [img[:, 50:300, :] for img in imgs] # type: ignore

                c,h,w = shape
                # c,h,w = 3, 180, 320
                # if zed in image, crop out x> 200
                if 'zed' in key:
                    imgs= [img[:, :400, :] for img in imgs] # type: ignore
                
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in imgs] # type: ignore
                # apply color jittering

                resize_imgs = [Image.fromarray(img) for img in resize_imgs]
                resize_imgs = [color_jitter(img) for img in resize_imgs]
                # apply random crop
                # resize_imgs = [random_crop(img) for img in resize_imgs]
                resize_imgs = [np.array(img) for img in resize_imgs]
                
                if 'dt' in key:
                    # perform image difference
                    resize_imgs = [compute_densetact_image_difference(resize_imgs[0], resize_imgs[i], threshold=70) for i in range(len(resize_imgs))]

                imgs = np.stack(resize_imgs, axis=0)
                # assert imgs[0].shape == (h,w,c)
                if is_fruit:
                    imgs = imgs[:-1]
                rgb_data_dict[key].append(imgs)

            # go through any depth image keys
            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                imgs = file['observations']['images'][key][()]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in imgs]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h,w,c)
                depth_data_dict[key].append(imgs)
            
            episode_points = []
            episode_colors = []

            # commented out 
            for key in touch_rgb_keys:
                assert key == 'densetact_images' 
                dt_names = ['dt_1', 'dt_2', 'dt_3']#, 'dt_4']
                all_dt_data = []
                for dt_name in dt_names:
                    imgs = file[dt_name][()]
                    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
                    imgs = np.stack(imgs, axis=0)

                    all_dt_data.append(imgs)

                episode_touch_images = np.stack(all_dt_data, axis=1)

                # assert episode_touch_images_stacked[0].shape == (h,w,c)
                if key not in touch_rgb_data_dict:
                    touch_rgb_data_dict[key] = list()
                touch_rgb_data_dict[key].append(episode_touch_images)
                # combine point clouds (later on)
                # Use finger pose and get in robot base frame
            
            
            # Go through eachstep in the episode
            if 'densetact_images' in touch_rgb_data_dict:
                all_points = []
                for i in range(episode_length):
                    all_points.append([])
                    episode_points.append([])
                    episode_colors.append([])
                    pose = lowdim_data_dict['robot_pose'][-1][i]  # get pose at step
                    touch_img = touch_rgb_data_dict['densetact_images'][-1][i]  # get touch image at step
                    angles = pose[9:25]  # Extract joint angles
                    ee_pos = pose[:3]    # Extract end-effector position
                    ee_rot = pose[3:9]   # Extract end-effector rotation (6D)
                    
                    # Get global transformation of wrist
                    rot_mat = tf.forward(ee_rot).reshape(3,3)
                    correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # to align hand coords to ee pose
                    T_base_to_wrist = np.eye(4)
                    T_base_to_wrist[:3, :3] = rot_mat @ correction
                    T_base_to_wrist[:3, 3] = ee_pos
                    
                    # # Shift from hand base to wrist (identical to policy_node.py)
                    # x_shift = 0.0131
                    # y_shift = 0.0375
                    # z_shift = 0.046

                    # T_hand_base_to_wrist = np.array([
                    #     [1.0, 0.0, 0.0, x_shift],
                    #     [0.0, 1.0, 0.0, y_shift],
                    #     [0.0, 0.0, 1.0, z_shift],
                    #     [0.0, 0.0, 0.0, 1.0]
                    # ])
                    # # T_base_to_wrist = np.dot(T_base_to_wrist, T_hand_base_to_wrist)
                    # T_base_to_wrist = np.dot(T_hand_base_to_wrist, T_base_to_wrist)


                    # Process each tactile image with the corresponding generator
                    for j, finger in enumerate(["thumb", "pointer", "middle"]):
                        if j >= len(touch_img):
                            continue  # Skip if image not available
                            
                        # Get the tactile image for this finger
                        def_image = touch_img[j]
                        
                        # Convert tactile image from (H,W,C) to (C,H,W) format if needed
                        if def_image.ndim == 3 and def_image.shape[2] == 3:
                            # Convert to CV2-friendly format (H,W,C)
                            def_image_cv = def_image.copy()
                        else:
                            # Assume it's already in the right format
                            def_image_cv = def_image.transpose(1, 2, 0)
                            
                        # Use the appropriate generator for this finger
                        generator = pcl_generators.get(finger)
                        if generator is None:
                            continue
                            
                        # Process the tactile image to get pointcloud
                        results = generator.process_tactile_image(
                            def_image_cv,
                            'stress1',  # Use stress1 for coloring
                            joint_state=angles,
                            ee_pos=None,
                            ee_rot=None
                        )

                        
                        # Extract the pointcloud (already transformed to global frame)
                        if results and 'pointcloud' in results and results['pointcloud'] is not None:
                            pointcloud = results['pointcloud']
                            
                            points = pointcloud[:, :3]  # XYZ coordinates
                            colors = pointcloud[:, 3:] if pointcloud.shape[1] > 3 else np.ones((len(points), 3))  # RGB colors
                            points,colors = downsample_with_fps(points, colors, None, num_points=100, output_points_colors=True) #downsample to 100 points subject to change

                            
                            # Apply transformation to the pointcloud
                            points = np.dot(points, T_base_to_wrist[:3, :3].T) + T_base_to_wrist[:3, 3]
                            # Min Max scaling for colors
                            # color_min = colors.min(axis=0)               # shape (n_features,)
                            # color_max = colors.max(axis=0)
                            # colors = (colors - color_min) / (color_max - color_min + 1e-8)  # shape (n_samples, n_features)
                            
                            # print(i)
                            # if i == 100:
                            #     import pdb; pdb.set_trace()
                                # visualize the point cloud
                                # visualize_point_cloud(points, colors)

                            # save points as npy
                            # np.save(f"/home/user/Documents/DT-Diffusion-Policy/gendp/gendp/pointcloud_{i}_{finger}.npy", points)
                            # save colors as npy
                            # np.save(f"colors_{i}_{finger}.npy", colors)
                            # Add to combined lists
                            all_points[i].extend(points.tolist())
                            # if i == 100:
                            #     import pdb; pdb.set_trace()
                            episode_points[i].extend(points.tolist())
                            episode_colors[i].extend(colors.tolist())
            
            for key in spatial_keys:
                # construct inputs for spatial fields processing
                combined = None
                view_keys = shape_meta['obs'][key]['info']['view_keys']
                distill_dino = shape_meta['obs'][key]['info']['distill_dino'] if 'distill_dino' in shape_meta['obs'][key]['info'] else False

                try:
                    color_seq = np.stack([file['image'][()] for k in view_keys], axis=1) # type: ignore
                except:
                    color_seq = np.stack([file['zed_rgb'][()] for k in view_keys], axis=1) # type: ignore
                color_seq = color_seq[..., :3]
                

                try:
                    depth_seq = np.stack([file['depth'][()] for k in view_keys], axis=1) # (T, V, H ,W)
                    extri_seq = np.stack([file['extrinsics'][()] for k in view_keys], axis=1) # (T, V, 4, 4)

                    intri_seq = np.stack([file['intrinsics'][()] for k in view_keys], axis=1) # (T, V, 3, 3)
                except:
                    depth_seq = np.stack([file['zed_depth_images'][()] for k in view_keys], axis=1)

                    fx = 1069.73
                    fy = 1069.73
                    cx = 1135.86
                    cy = 680.69

                    single_intrinsic_matrix = np.array([
                        [fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]
                    ])

                    # make single intrinsic matrix for all views
                    intri_seq = np.stack([single_intrinsic_matrix for k in view_keys], axis=0)
                    # make multiple intrinsic matrices for the amount of steps
                    intri_seq = np.stack([intri_seq for k in range(color_seq.shape[0])], axis=0)

                    single_extrinsic_matrix = np.array([
                                                        [0.0200602,  0.7492477, -0.6619859,  0.580],
                                                        [0.9983952,  0.0200602,  0.0529589,  0.020],
                                                        [0.0529589, -0.6619859, -0.7476429,  0.570],
                                                        [0.000,      0.000,      0.000,      1.000]
                                                    ])
                    extri_seq = np.stack([single_extrinsic_matrix for k in view_keys], axis=0)
                    extri_seq = np.stack([extri_seq for k in range(color_seq.shape[0])], axis=0)
 

                calibrated = shape_meta['obs'][key]['info']['calibrated'] if 'calibrated' in shape_meta['obs'][key]['info'] else False
                if not calibrated:
                    all_points = []

                episode_semantic_features = []
                episode_semantic_fields = []
                color_jittered_imgs = []
                distill_dino = False

                for i in tqdm(range(color_seq.shape[0]), desc="Processing Frames with VFM Featurizer..."):
                    x = color_seq[i][0]

                    # rgb to bgr
                    x = x[..., [2, 1, 0]]

                    # perform some color jitter on the image
                    img = Image.fromarray(x)
    
                    # Apply color jitter with specified parameters
                    color_jitter = transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.02
                    )

                    jittered_img = np.array(color_jitter(img))
    
                    # Convert back to numpy array
                    color_seq[i] = np.array(jittered_img)

                    color_jittered_imgs.append(jittered_img)

                    # this determines if we want to compute the part level semantics 
                    # focusing exclusively on objects
                    part_level_semantics = False

                    # get masks for the important stuff like the ball and robot hand
                    if distill_dino:
                        # get dino features with pca, 3 dimensions (this could be changed later on)
                        semantic_features = radio_model.featurize(jittered_img, perform_pca_single_image=False)

                        # torch clear cache
                        torch.cuda.empty_cache()
                        # add raw features to the episode!
                        episode_semantic_features.append(semantic_features[0])

                    if part_level_semantics:
                        # get the features of each part
                        results = object_semantics.get_semantics(jittered_img)

                        semantic_field = results['max_similarity'][0]  # Remove batch dimension

                        semantic_field = (semantic_field - semantic_field.min()) / (semantic_field.max() - semantic_field.min())
        
                        # Create heatmap
                        semantic_field = semantic_field.cpu().numpy()
                        semantic_field = cv2.resize(semantic_field, (jittered_img.shape[1], jittered_img.shape[0]))

                        # Apply a colormap to convert grayscale to RGB
                        # COLORMAP_JET is a common choice for heatmaps, but you can use others like
                        # COLORMAP_VIRIDIS, COLORMAP_INFERNO, COLORMAP_PLASMA, etc.
                        heatmap = cv2.applyColorMap((semantic_field * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        episode_semantic_fields.append(heatmap)

                color_jittered_imgs = np.array(color_jittered_imgs)

                # perform PCA on the episode
                if distill_dino:
                    episode_semantic_features = np.stack(episode_semantic_features, axis=0) # (T, 1024)
                    episode_semantic_features = radio_model.perform_batch_pca(episode_semantic_features, 
                                                                           jittered_img.shape[0], jittered_img.shape[1])

                # go through each frame in the episode
                all_masks = []
                for i in tqdm(range(color_jittered_imgs.shape[0]), desc="Processing Frames"):
                    color_img = color_jittered_imgs[i]

                    if distill_dino:
                        semantic_features = episode_semantic_features[i] # type: ignore

                    # make depths nans if beyond 1 meter
                    depth_seq[i, 0][depth_seq[i, 0] > 1.1] = np.nan
                    # depth_seq[i, 0][depth_seq[i, 0] > 0.75] = np.nan

                    n = 200
                    # Set the pixels with x <= n to NaN
                    # depth_seq[i, 0][:, :n] = np.nan

                    # get masks for the important stuff like the ball and robot hand
                    masks, if_init = get_gsam2_masks(sam2_predictor, grounding_model, color_img, if_init)

                    # next steps: mask out the features of the relevant objects, THEN perform PCA.

                    # construct one image from all masks
                    main_mask = np.zeros((color_img.shape[0], color_img.shape[1]), dtype=np.uint8)
                    for mask in masks[:-1]:
                        main_mask = np.logical_or(main_mask, mask)

                    all_masks.append(main_mask)
                    
                    full_points = None

                    if distill_dino:
                        semantic_features = (semantic_features * 255).astype(np.uint8)  # type: ignore
                        points, colors, _ = compute_point_cloud(
                            semantic_features, depth_seq[i, 0], intri_seq[i, 0], extri_seq[i, 0], masks
                        )
                        points, colors = crop_points_within_bounds(points, colors)

                    else:
                        # TODO: construct two separate point clouds for each object, then make sure the object has enough points!
                        points, colors, _ = compute_point_cloud(
                        color_img, depth_seq[i, 0], intri_seq[i, 0], extri_seq[i, 0], masks[1:]
                        )

                        # create PC of the robot 
                        fruit_points, fruit_colors, _ = compute_point_cloud(
                            color_img, depth_seq[i, 0], intri_seq[i, 0], extri_seq[i, 0], [masks[0]]
                        )

                        points, colors = crop_points_within_bounds(points, colors)
                        fruit_points, fruit_colors = crop_points_within_bounds(fruit_points, fruit_colors)

                    # downsample based on other available points
                    if not calibrated:
                        fruit_points, fruit_colors = downsample_with_fps(fruit_points, fruit_colors, None, num_points=max_pts_num//2, output_points_colors=True)

                        points, colors = downsample_with_fps(points, colors, None, num_points=max_pts_num - len(fruit_points), output_points_colors=True)

                        # combine 
                        points = np.concatenate([points, fruit_points], axis=0)
                        colors = np.concatenate([colors, fruit_colors], axis=0)

                        
                    else:
                        points, colors = downsample_with_fps(points, colors, None, num_points=max_pts_num-len(all_points[i]),
                                                                        output_points_colors=True)

                    combined_points = np.concatenate([points, colors], axis=-1)

                    # visualize the point cloud
                    if not calibrated:
                        # no calibration, just use the points from vision
                        episode_points.append(points)
                        episode_colors.append(colors)

                        channels = shape_meta['obs'][key]['shape'][0]

                        if channels == 6:
                            all_points = all_points + [combined_points]
                        else:
                            all_points = all_points + [points]
                    else:
                        # calibration is done, add the points from vision
                        modified_colors = []
                        # for each color point, add an extra 0 to the end
                        for color in colors:
                            r, g, b = color[:3]
                            # add a 0 to mark a non touched object
                            color = np.array([r, g, b, 0])
                            modified_colors.append(color)

                        modified_colors = np.array(modified_colors)

                        episode_points[i].extend(points)
                        episode_colors[i].extend(colors)
                        
                        all_points[i].extend(points)


                # visualize all masks in video (2d)
                # pause_time = 0.2
                # plt.figure(figsize=(8, 8))
    
                # while True:  # Infinite loop - use Ctrl+C to stop
                #     for i, mask in enumerate(all_masks):
                #         plt.clf()  # Clear the figure
                        
                #         if mask.ndim == 2:
                #             plt.imshow(mask, cmap='gray')
                #         else:
                #             plt.imshow(mask)
                            
                #         plt.title(f'Mask {i+1}/{len(all_masks)}')
                #         plt.axis('off')
                #         plt.tight_layout()
                #         plt.draw()
                #         plt.pause(pause_time)  # Pause between frames

                if remove_last_step:
                    all_points = all_points[:-1]
                    
                if key not in spatial_data_dict:
                    spatial_data_dict[key] = list()
                spatial_data_dict[key] = spatial_data_dict[key] + [np.array(all_points)]
                # feats_per_epi = feats_per_epi + aggr_feats_ls


            # visualize point cloud as video
            vis = o3d.visualization.Visualizer()
            vis.create_window(height=960, width=1280)
            pcd = o3d.geometry.PointCloud()
            points = np.random.rand(10, 3)
            pcd.points = o3d.utility.Vector3dVector(points)

            vis.add_geometry(pcd)

            end_effector_pcd = o3d.geometry.PointCloud()
            end_effector_point = np.array([[0, 0, 0]])  # Replace with actual end effector coordinates
            end_effector_color = np.array([[1.0, 0.0, 0.0]])  # Red color

            end_effector_pcd.points = o3d.utility.Vector3dVector(end_effector_point)
            end_effector_pcd.colors = o3d.utility.Vector3dVector(end_effector_color)
            vis.add_geometry(end_effector_pcd)
            for idx, val in enumerate(episode_points):
                pcd.points = o3d.utility.Vector3dVector(episode_points[idx])
                pcd.colors = o3d.utility.Vector3dVector(episode_colors[idx])
                # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=lowdim_data_dict['robot_pose'][-1][idx][:3])
                # vis.add_geometry(coordinate_frame)
                vis.update_geometry(pcd)
                keep_running = vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)
            
            # kill the window
            vis.destroy_window()



    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as _:
            return False
    if only_pose:
        lowdim_keys = ['pose']
    # dump data_dict


    print('Dumping meta data')
    n_steps = episode_ends[-1]
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)

    print('Dumping lowdim data')
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype
        )
    
    print('Dumping rgb data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint8
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
    
    print('Dumping depth data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint16
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
            
    # dump spatial data
    print('Dumping spatial data')
    for key, data in spatial_data_dict.items():
        # pad to max_pts_num
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    print('Dumping touch image data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for key, data in touch_rgb_data_dict.items():
            data = np.concatenate(data, axis=0)
            _ = data_group.array(
                name=key,
                data=data,
                shape=data.shape,
                chunks=(1,) + data.shape[1:],
                compressor=None,
                dtype=data.dtype
            )

            # shape = tuple(shape_meta['obs'][key]['shape'])
            shape = (3, 256, 256)
            c,h,w = shape
    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def display_images_tkinter(images, fps=5, window_width=800, window_height=600):
    """
    Display images using tkinter with larger size
    
    Args:
        images: List of images as numpy arrays
        fps: Frames per second (default: 5)
        window_width: Preferred window width (default: 800)
        window_height: Preferred window height (default: 600)
    """
    delay_ms = int(1000 / fps)  # Delay in milliseconds
    
    # Create the main window
    root = tk.Tk()
    root.title("Image Sequence")
    
    # Set window size and position
    root.geometry(f"{window_width}x{window_height}+100+100")
    
    # Make window resizable
    root.resizable(True, True)
    
    # Create a frame for better layout
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Create a label to display the image with larger size
    image_label = tk.Label(frame)
    image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Frame counter label with larger font
    counter_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
    counter_label.pack(pady=5)
    
    # Add controls
    control_frame = tk.Frame(root)
    control_frame.pack(fill=tk.X, pady=5)
    
    # Variable to track if playback is paused
    paused = tk.BooleanVar(value=False)
    current_index = [0]  # Use list to make it mutable within nested functions
    
    # Keep references to the PhotoImage objects
    photo_references = []
    
    def resize_image(img, target_width, target_height):
        """Resize image while maintaining aspect ratio"""
        width, height = img.size
        ratio = min(target_width/width, target_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def update_image(index=None):
        if index is not None:
            current_index[0] = index
            
        if current_index[0] >= len(images):
            current_index[0] = 0  # Loop back to the beginning
            
        if paused.get():
            return
            
        # Convert numpy array to PIL Image
        if isinstance(images[current_index[0]], np.ndarray):
            img = Image.fromarray(images[current_index[0]])
        else:
            img = images[current_index[0]]
        
        # Get available space in the window
        available_width = image_label.winfo_width() or window_width - 20
        available_height = image_label.winfo_height() or window_height - 100
        
        # Resize image to fit the window while maintaining aspect ratio
        img_resized = resize_image(img, available_width, available_height)
            
        # Convert to PhotoImage for tkinter
        photo = ImageTk.PhotoImage(img_resized)
        photo_references.clear()  # Clear previous references
        photo_references.append(photo)  # Keep a reference
        
        # Update the image and counter
        image_label.configure(image=photo)
        counter_label.configure(text=f"Frame: {current_index[0]+1}/{len(images)}")
        
        # Schedule the next update
        if not paused.get():
            current_index[0] += 1
            root.after(delay_ms, update_image)
    
    def toggle_pause():
        paused.set(not paused.get())
        if not paused.get():
            # Resume playback
            update_image()
            pause_button.config(text="Pause")
        else:
            pause_button.config(text="Play")
    
    def prev_frame():
        current_index[0] = max(0, current_index[0] - 2)  # -2 because update_image will add 1
        paused.set(True)
        pause_button.config(text="Play")
        update_image()
        
    def next_frame():
        current_index[0] = min(len(images) - 1, current_index[0])
        paused.set(True)
        pause_button.config(text="Play")
        update_image()
    
    # Create control buttons
    prev_button = tk.Button(control_frame, text="Previous", command=prev_frame)
    prev_button.pack(side=tk.LEFT, padx=10)
    
    pause_button = tk.Button(control_frame, text="Pause", command=toggle_pause)
    pause_button.pack(side=tk.LEFT, padx=10)
    
    next_button = tk.Button(control_frame, text="Next", command=next_frame)
    next_button.pack(side=tk.LEFT, padx=10)
    
    # Add a slider for frame selection
    frame_slider = tk.Scale(root, from_=1, to=len(images), orient=tk.HORIZONTAL, 
                            label="Frame Selection", command=lambda v: update_image(int(v)-1))
    frame_slider.pack(fill=tk.X, padx=20, pady=10)
    
    # Function to handle window resize
    def on_resize(event):
        if not paused.get():
            return  # Don't update during continuous playback to avoid flickering
        # Update the image to fit the new window size
        update_image(current_index[0])
    
    # Bind the resize event
    root.bind("<Configure>", on_resize)
    
    # Start the update loop
    root.after(100, update_image)  # Short delay to ensure window is fully created
    
    # Start the tkinter main loop
    root.mainloop()
    
    # Clean up
    root.destroy()

    
class RealDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_dir: str,
            vis_input: False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=True,
            use_cache=True,
            seed=42,
            val_ratio=0.0,
            manual_val_mask=False,
            manual_val_start=-1,
            n_obs_steps=None,
            robot_name='panda',
            expected_labels=None,
            exclude_colors=[]
            ):
        super().__init__()
        # rotation_transformer = RotationTransformer(
        #     from_rep='euler_angles', to_rep=rotation_rep, from_convention='xyz')
        rotation_transformer = RotationTransformer(
            from_rep='quaternion', to_rep=rotation_rep)#, from_convention='xyz')
        

        replay_buffer = None
        fusion = None
        cache_info_str = ''
        for key, attr in shape_meta['obs'].items():
            if ('type' in attr) and (attr['type'] == 'depth'):
                cache_info_str += '_rgbd'
                break
        if 'd3fields' in shape_meta['obs']:
            use_seg = False
            distill_dino = shape_meta['obs']['d3fields']['info']['distill_dino'] if 'distill_dino' in shape_meta['obs']['d3fields']['info'] else False
            if use_seg:
                cache_info_str += '_seg'
            else:
                cache_info_str += '_no_seg'
            if not distill_dino:
                cache_info_str += '_no_dino'
            elif distill_dino:
                cache_info_str += '_distill_dino'
            else:
                cache_info_str += '_dino'
            if 'key' in shape_meta['action'] and shape_meta['action']['key'] == 'joint_action':
                cache_info_str += '_joint'
            else:
                cache_info_str += '_eef'
        if use_cache:
            cache_zarr_path = os.path.join(dataset_dir, f'cache{cache_info_str}.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # create fusion if necessary
                    self.fusion_dtype = torch.float16
                    # If get stuck here, try to `export OMP_NUM_THREADS=1`
                    # refer: https://github.com/pytorch/pytorch/issues/21956
                    for key, attr in shape_meta['obs'].items():
                        if ('type' in attr) and (attr['type'] == 'spatial'):
                            num_cam = len(attr['info']['view_keys'])
                            break
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_dir=dataset_dir, 
                            rotation_transformer=rotation_transformer,
                            fusion=fusion,
                            robot_name=robot_name,
                            expected_labels=expected_labels,
                            exclude_colors=exclude_colors,
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            # create fusion if necessary
            self.fusion_dtype = torch.float16
            # If get stuck here, try to `export OMP_NUM_THREADS=1`
            # refer: https://github.com/pytorch/pytorch/issues/21956
            for key, attr in shape_meta['obs'].items():
                if ('type' in attr) and (attr['type'] == 'spatial'):
                    num_cam = len(attr['info']['view_keys'])
                    break
            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_dir=dataset_dir,
                rotation_transformer=rotation_transformer,
                fusion=fusion,
                robot_name=robot_name,
                expected_labels=expected_labels,
            )
        self.replay_buffer = replay_buffer
        
        if vis_input:
            if 'd3fields' in shape_meta['obs'] and shape_meta['obs']['d3fields']['info']['distill_dino']:
                vis_distill_feats = True
            else:
                vis_distill_feats = False
            self.replay_buffer.visualize_data(output_dir=os.path.join(dataset_dir, f'replay_buffer_vis_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}'), vis_distill_feats=vis_distill_feats)
        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        spatial_keys = list()
        touch_rgb_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'spatial':
                spatial_keys.append(key)
            elif type == 'touch_rgb':
                touch_rgb_keys.append(key)

        if 'info' in shape_meta['action'] and 'only_pose' in shape_meta['action']['info']:
            if shape_meta['action']['info']['only_pose']:
                lowdim_keys.append('pose') 
        

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed)
        else:
            try:
                assert manual_val_start >= 0
                assert manual_val_start < replay_buffer.n_episodes
            except:
                raise RuntimeError('invalid manual_val_start')
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys + spatial_keys + touch_rgb_keys:
                key_first_k[key] = n_obs_steps
            

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            dataset_dir=dataset_dir,
            key_first_k=key_first_k,
            shape_meta=shape_meta,)
        
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.spatial_keys = spatial_keys
        self.touch_rgb_keys = touch_rgb_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_dir = dataset_dir
        self.key_first_k = key_first_k

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            dataset_dir=self.dataset_dir,
            key_first_k=self.key_first_k,
            shape_meta=self.shape_meta,
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.use_legacy_normalizer:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        else:
            raise RuntimeError('unsupported')
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('vel'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == 'prev_robot_pose':
                #for the total state, we don't need to normalize since everything is in [-1,1] (I think)
                this_normalizer = get_identity_normalizer_from_stat(stat) 
            elif key == 'robot_pose':
                this_normalizer = normalizer_from_stat(stat)
            elif key == 'pose':
                #for the total state, we don't need to normalize since everything is in [-1,1] (I think)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == 'robot_pos_ang':
                dims_to_skip = np.zeros(stat['mean'].shape, dtype=np.bool)
                dims_to_skip[0:9] = True
                this_normalizer = get_range_normalizer_from_stat(stat, ignore_dim=dims_to_skip)
            else:
                raise RuntimeError('unsupported')

            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_range_normalizer()
        
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
        
        # spatial
        is_joint = ('key' in self.shape_meta['action'].keys()) and (self.shape_meta['action']['key'] == 'joint_action')
        for key in self.spatial_keys:
            B, N, C = self.replay_buffer[key].shape
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            if self.shape_meta['obs'][key]['shape'][0] == 1027:
                # compute normalizer for feats of top 10 demos
                feats = []
                for demo_i in range(1):
                    if is_joint:
                        feats_prefix += '_joint'
                    if os.path.exists(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')):
                        with h5py.File(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')) as file:
                            feats.append(file['feats'][()])
                feats = np.concatenate(feats, axis=0) # (10, T, N, 1024)
                feats = feats.reshape(-1, 1024) # (10 * T * N, 1024)
                feat_stat = array_to_stats(feats)
                for stat_key in stat:
                    stat[stat_key] = np.concatenate([stat[stat_key], feat_stat[stat_key]], axis=0)
                normalizer[key] = get_range_normalizer_from_stat(stat, ignore_dim=[0,1,2])
            else:
                normalizer[key] = get_identity_normalizer_from_stat(stat)
        
        for key in self.touch_rgb_keys:
            stat = {
                'min': np.array([0], dtype=np.float32),
                'max': np.array([1], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32),
                'std': np.array([np.sqrt(1/12)], dtype=np.float32)
            }

            normalizer[key] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 1000.
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
        for key in self.spatial_keys:
            obs_dict[key] = np.moveaxis(sample[key][T_slice],1,2).astype(np.float32)
            del sample[key]
        for key in self.touch_rgb_keys:
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,2).astype(np.float32)
            del sample[key]

        data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(sample['action'].astype(np.float32))
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data

if __name__ == '__main__':
    pass