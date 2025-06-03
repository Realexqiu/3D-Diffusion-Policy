from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Bool
import time
import cv2
import numpy as np
import torch
import pathlib
from pathlib import Path
from typing import Optional, Tuple, Union, Mapping, Any
import hydra
import diffusers
import dill
import zarr
from typing import Dict, Any, Mapping, Optional, Tuple, Union

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from rotation_transformer import RotationTransformer
from cv_bridge import CvBridge


class InferenceUtils:

    def init_policy_node(self, model_path: str, shared_obs, action_queue, start_time, sync_queue_size=100, sync_slop=0.05, inference_rate=30):
        """Initialize all the ROS2 components for the policy node.
        
        This method sets up:
        - Policy and shape metadata
        - Subscribers based on available sensors
        - Publishers for robot control
        - Buffers and synchronizers
        - Internal state variables
        """
        np.set_printoptions(suppress=True, precision=4)
        
        # Store shared resources
        self.shared_obs = shared_obs
        self.action_queue = action_queue
        self.start_time = start_time
        
        # Initialize QoS profile
        self.sensor_qos = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST
        )
        
        # Load metadata
        self._load_metadata(model_path)
        
        # Initialize subscribers and buffers
        self._initialize_subscribers(sync_queue_size, sync_slop)
        
        # Initialize publishers
        self._initialize_publishers()
        
        # Initialize other components
        self._initialize_components()
        
        # Create timers
        self.create_timer(1/inference_rate, self.timer_callback)
        self.create_timer(1/inference_rate, self.update_observation)

        # Jiggle gripper to activate it
        msg = Float32()
        msg.data = 0.05
        self.gripper_pub.publish(msg)
        time.sleep(0.5)
        msg.data = 0.0
        self.gripper_pub.publish(msg)
        self.gripper_state = 0.0

        # Reset robot to home position
        self.reset_xarm_pub.publish(Bool(data=True))

    def _load_metadata(self, model_path: str):
        """Load policy and shape metadata from model."""
        self.shape_meta = load_shape_meta(model_path)
        self.policy_meta = self.load_policy_meta(model_path)
        
        # Determine available observation keys
        self.obs_keys = list(self.shape_meta['obs'].keys())
        self.has_zed_rgb = 'zed_rgb' in self.obs_keys
        self.has_zed_depth = 'zed_depth_images' in self.obs_keys
        self.has_rs_side_rgb = 'rs_side_rgb' in self.obs_keys
        self.has_rs_wrist_rgb = 'rs_wrist_rgb' in self.obs_keys
        
        # Log configuration
        self.get_logger().info(f"Policy observation keys: {self.obs_keys}")
        self.get_logger().info(f"Using ZED RGB camera: {self.has_zed_rgb}")
        self.get_logger().info(f"Using ZED Depth camera: {self.has_zed_depth}")
        self.get_logger().info(f"Using RealSense RGB Side Camera: {self.has_rs_side_rgb}")
        self.get_logger().info(f"Using RealSense RGB Wrist Camera: {self.has_rs_wrist_rgb}")

    def _initialize_subscribers(self, sync_queue_size, sync_slop):
        """Initialize subscribers based on available sensors."""
        # Core subscribers (always needed)
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, '/robot_pose', qos_profile=self.sensor_qos)
        self.gripper_sub = message_filters.Subscriber(self, Float32, '/gripper_state', qos_profile=self.sensor_qos)
        
        subscribers = [self.pose_sub, self.gripper_sub]
        
        # Initialize buffers
        self.pose_buffer = []
        
        # Conditional subscribers and buffers
        if self.has_zed_rgb:
            self.zed_rgb_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/zed_image/rgb/compressed', qos_profile=self.sensor_qos)
            subscribers.append(self.zed_rgb_compressed_sub)
            self.zed_rgb_buffer = []
            self.zed_meta = get_cam_preprocess(self.policy_meta, 'zed_rgb')
            
        if self.has_zed_depth:
            self.zed_depth_sub = message_filters.Subscriber(self, Image, '/zed_image/depth', qos_profile=self.sensor_qos)
            subscribers.append(self.zed_depth_sub)
            self.zed_depth_buffer = []
            
        if self.has_rs_side_rgb:
            self.rs_side_color_compressed_sub = message_filters.Subscriber(
                self, CompressedImage, 
                '/rs_side/rs_side/color/image_raw/compressed', 
                qos_profile=self.sensor_qos
            )
            subscribers.append(self.rs_side_color_compressed_sub)
            self.rs_side_rgb_buffer = []
            self.rs_side_meta = get_cam_preprocess(self.policy_meta, 'rs_side_rgb')
            
        if self.has_rs_wrist_rgb:
            self.rs_wrist_color_compressed_sub = message_filters.Subscriber(
                self, CompressedImage, 
                '/rs_wrist/rs_wrist/color/image_raw/compressed', 
                qos_profile=self.sensor_qos
            )
            subscribers.append(self.rs_wrist_color_compressed_sub)
            self.rs_wrist_rgb_buffer = []
            self.rs_wrist_meta = get_cam_preprocess(self.policy_meta, 'rs_wrist_rgb')
        
        # Create synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, 
            queue_size=sync_queue_size,
            slop=sync_slop,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_obs_callback)

    def _initialize_publishers(self):
        """Initialize all publishers."""
        self.gripper_pub = self.create_publisher(Float32, '/gripper_position', 1)
        self.reset_xarm_pub = self.create_publisher(Bool, '/reset_xarm', 1)
        self.pause_xarm_pub = self.create_publisher(Bool, '/pause_xarm', 1)
        self.pub_robot_pose = self.create_publisher(PoseStamped, '/xarm_position', 1)

    def _initialize_components(self):
        """Initialize other components."""
        self._bridge = CvBridge()
        self.observation_horizon = 2
        self.tf = RotationTransformer('rotation_6d', 'quaternion')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paused = False
        self.even = True
        self.pending_actions = []
        self.gripper_state = 0.0

    def load_policy_meta(self, model_path: str) -> Dict:
        """Load policy metadata from checkpoint."""
        path = pathlib.Path(model_path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        return payload["cfg"].policy.meta if hasattr(payload["cfg"].policy, "meta") else {}
    
    def synced_obs_callback(self, *args):
        """Process synchronized observations and generate point cloud"""
        
        # self.get_logger().info("Synced obs callback")
        
        # Parse arguments based on what's available
        arg_idx = 0
        pose_msg = args[arg_idx]
        arg_idx += 1
        
        gripper_msg = args[arg_idx]
        arg_idx += 1
        
        zed_rgb_msg = None
        zed_depth_msg = None
        rs_side_msg = None
        rs_wrist_msg = None
        
        if self.has_zed_rgb:
            zed_rgb_msg = args[arg_idx]
            arg_idx += 1
            
        if self.has_rs_side_rgb:
            rs_side_msg = args[arg_idx]
            arg_idx += 1

        if self.has_rs_wrist_rgb:
            rs_wrist_msg = args[arg_idx]
            arg_idx += 1

        # Update gripper state
        self.gripper_state = gripper_msg.data
        
        # Process pose message
        self.pose_callback(pose_msg)

        # Process camera messages based on availability
        if self.has_zed_rgb and zed_rgb_msg is not None:
            self.zed_rgb_compressed_callback(zed_rgb_msg)

        if self.has_zed_depth and zed_depth_msg is not None:
            self.zed_depth_callback(zed_depth_msg)

        if self.has_rs_side_rgb and rs_side_msg is not None:
            self.rs_side_rgb_compressed_callback(rs_side_msg)

        if self.has_rs_wrist_rgb and rs_wrist_msg is not None:
            self.rs_wrist_rgb_compressed_callback(rs_wrist_msg)

        self.update_observation()

    def update_observation(self):
        """Consolidate the latest *horizon* observations and push to `shared_obs`."""
        # Check minimum buffer lengths based on what cameras are available
        buffer_lengths = [len(self.pose_buffer)]
        if self.has_zed_rgb:
            buffer_lengths.append(len(self.zed_rgb_buffer))
        if self.has_rs_side_rgb:
            buffer_lengths.append(len(self.rs_side_rgb_buffer))
        if self.has_rs_wrist_rgb:
            buffer_lengths.append(len(self.rs_wrist_rgb_buffer))
            
        min_len = min(buffer_lengths)
        if min_len < self.observation_horizon:
            return  # not enough data yet

        # Robot Pose
        pose_slice = self.pose_buffer[-self.observation_horizon:]
        pose_np = np.stack([p[0] for p in pose_slice])           # (T, 10)
        pose_tstamps = np.array([p[1] for p in pose_slice])           # (T,)
        pose_tensor = torch.from_numpy(pose_np).unsqueeze(0)         # (1, T, 10)

        # Build observation dict dynamically
        obs_dict = {
            'pose': pose_tensor,
            'pose_timestamps': pose_tstamps,
        }

        # Add ZED data if available
        if self.has_zed_rgb:
            zed_rgb_slice = self.zed_rgb_buffer[-self.observation_horizon:]
            zed_rgb_np = np.stack([r[0] for r in zed_rgb_slice])            # (T, 3, H, W)
            zed_rgb_tstamps = np.array([r[1] for r in zed_rgb_slice])            # (T,)
            zed_rgb_tensor = torch.from_numpy(zed_rgb_np).unsqueeze(0)          # (1, T, 3, H, W)
            obs_dict['zed_rgb'] = zed_rgb_tensor
            obs_dict['zed_rgb_timestamps'] = zed_rgb_tstamps

        # Add RealSense data if available
        if self.has_rs_side_rgb:
            rs_side_slice = self.rs_side_rgb_buffer[-self.observation_horizon:] 
            rs_side_np = np.stack([r[0] for r in rs_side_slice])             # (T, 3, H, W)
            rs_side_rgb_tstamps = np.array([r[1] for r in rs_side_slice])             # (T,)
            rs_side_rgb_tensor = torch.from_numpy(rs_side_np).unsqueeze(0)           # (1, T, 3, H, W)
            obs_dict['rs_side_rgb'] = rs_side_rgb_tensor
            obs_dict['rs_side_rgb_timestamps'] = rs_side_rgb_tstamps

        if self.has_rs_wrist_rgb:
            rs_wrist_slice = self.rs_wrist_rgb_buffer[-self.observation_horizon:] 
            rs_wrist_np = np.stack([r[0] for r in rs_wrist_slice])             # (T, 3, H, W)
            rs_wrist_tstamps = np.array([r[1] for r in rs_wrist_slice])             # (T,)
            rs_wrist_tensor = torch.from_numpy(rs_wrist_np).unsqueeze(0)           # (1, T, 3, H, W)
            obs_dict['rs_wrist_rgb'] = rs_wrist_tensor
            obs_dict['rs_wrist_rgb_timestamps'] = rs_wrist_tstamps

        # Push to shared memory (IPC)
        self.shared_obs['obs'] = obs_dict

    def pose_callback(self, msg):
        """Process robot pose"""
        robot_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] 

        # tf.inverse expects [wxyz]
        robot_ori = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] 
        
        # Convert to 6D rotation representation
        robot_ori_tensor = torch.tensor(robot_ori, dtype=torch.float32)
        robot_ori_6d = self.tf.inverse(robot_ori_tensor)

        # Combine position, orientation
        robot_pose = np.concatenate([robot_pos, robot_ori_6d.numpy(), [self.gripper_state]], axis=None)
        self.pose_buffer.append((robot_pose, time.monotonic() - self.start_time))
        if len(self.pose_buffer) > self.observation_horizon:
            self.pose_buffer.pop(0)

    def zed_rgb_compressed_callback(self, msg):
        """Process zed compressed rgb message"""
        zed_rgb_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        zed_rgb_img = zed_rgb_img[:, :, ::-1]

        # #         # Center crop settings
        # # ENABLE_CENTER_CROP_RS = False  # Set to False to disable center cropping
        # # RS_CROP_WIDTH = 850      # Target width for center crop
        # # RS_CROP_HEIGHT = 420     # Target height for center crop
        # # ENABLE_CENTER_CROP_ZED = True
        # ZED_CROP_WIDTH = 320
        # ZED_CROP_HEIGHT = 360

        # # # Off-center crop settings (set to None for center crop, or specify offset)
        # # # Positive values move crop towards bottom-right, negative towards top-left
        # # RS_CROP_OFFSET_X = None    # Horizontal offset from center (None for center crop)
        # # RS_CROP_OFFSET_Y = None    # Vertical offset from center (None for center crop)
        # ZED_CROP_OFFSET_X = -300   # For ZED camera
        # ZED_CROP_OFFSET_Y = None   # For ZED camera
        # ENABLE_CENTER_CROP_ZED = True

        #    # Apply cropping if enabled
        # if ENABLE_CENTER_CROP_ZED:
        #     zed_rgb_img = crop_frame(zed_rgb_img, (ZED_CROP_WIDTH, ZED_CROP_HEIGHT), 
        #                            offset_x=ZED_CROP_OFFSET_X, offset_y=ZED_CROP_OFFSET_Y)

        if self.zed_meta['crop']['enabled']:
            # import pdb; pdb.set_trace() # is this enabled?            
            zed_rgb_img = self.crop_frame(zed_rgb_img, self.zed_meta['crop']['size'], self.zed_meta['crop']['offset'])

    

        zed_rgb_img = self.resize_for_policy(zed_rgb_img, 'zed_rgb')

        self.zed_rgb_buffer.append((zed_rgb_img, time.monotonic() - self.start_time))
        if len(self.zed_rgb_buffer) > self.observation_horizon:
            self.zed_rgb_buffer.pop(0)

    def rs_side_rgb_compressed_callback(self, msg):
        """Process rs compressed rgb message"""
        rs_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rs_img = rs_img[:, :, ::-1]

        if self.rs_side_meta['crop']['enabled']:
            rs_img = self.crop_frame(rs_img, self.rs_side_meta['crop']['size'], self.rs_side_meta['crop']['offset'])

        rs_img = self.resize_for_policy(rs_img, 'rs_side_rgb')

        self.rs_side_rgb_buffer.append((rs_img, time.monotonic() - self.start_time))
        if len(self.rs_side_rgb_buffer) > self.observation_horizon:
            self.rs_side_rgb_buffer.pop(0)

    def rs_wrist_rgb_compressed_callback(self, msg):
        """Process rs compressed rgb message"""
        rs_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rs_img = rs_img[:, :, ::-1]

        if self.rs_wrist_meta['crop']['enabled']:
            rs_img = self.crop_frame(rs_img, self.rs_wrist_meta['crop']['size'], self.rs_wrist_meta['crop']['offset'])

        rs_img = self.resize_for_policy(rs_img, 'rs_wrist_rgb')

        self.rs_wrist_rgb_buffer.append((rs_img, time.monotonic() - self.start_time))
        if len(self.rs_wrist_rgb_buffer) > self.observation_horizon:
            self.rs_wrist_rgb_buffer.pop(0)

    def resize_for_policy(self, img: np.ndarray, cam_key: str) -> np.ndarray:
        """
        Resize *and* change layout from HWC-BGR (OpenCV) to CHW-RGB according to `shape_meta`.
        """
        C, H, W = self.shape_meta['obs'][cam_key]['shape']
        # self.get_logger().info(f"Resizing {cam_key} to {H}x{W}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR ➜ RGB
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))                  # HWC ➜ CHW
        assert img.shape == (C, H, W), \
            f"{cam_key} expected {(C,H,W)}, got {img.shape}"
        return img

    # ----------------------------------------------------------------------
    # Pygame helpers
    # ----------------------------------------------------------------------
    def reset_xarm(self):
        """Send the arm to a predefined home pose and fully open the gripper."""
        self.get_logger().info("Reset xarm.")

        # Home pose (position in metres; quaternion = [w, x, y, z])
        ee_pose = PoseStamped()
        ee_pose.header.stamp = self.get_clock().now().to_msg()
        ee_pose.pose.position.x = 0.1669
        ee_pose.pose.position.y = 0.0019
        ee_pose.pose.position.z = 0.2308
        w, x, y, z = 0.9999, -0.00995, 0.00507, 0.00785
        ee_pose.pose.orientation.w = w
        ee_pose.pose.orientation.x = x
        ee_pose.pose.orientation.y = y
        ee_pose.pose.orientation.z = z

        self.pub_robot_pose.publish(ee_pose)

        # Fully open the gripper
        self.gripper_pub.publish(Float32(data=0.0))

    def pause_policy(self):
        """Pause executor and flush any queued actions."""
        self.get_logger().info("Pause policy.")
        self.pause_xarm_pub.publish(Bool(data=True))
        self.paused = True
        self.shared_obs["paused"] = True
        self.pending_actions.clear()
        while not self.action_queue.empty():
            _ = self.action_queue.get_nowait()

    def resume_policy(self):
        """Resume execution after a pause."""
        self.get_logger().info("Resume policy.")
        self.pause_xarm_pub.publish(Bool(data=False))
        self.paused = False
        self.shared_obs["paused"] = False

    # ----------------------------------------------------------------------
    # Housekeeping
    # ----------------------------------------------------------------------
    def cleanup(self):
        """Resource-cleanup hook (override if needed)."""
        # Placeholder – add custom shutdown logic here.
        pass

    def generate_ee_action_msg(self, action: torch.Tensor) -> PoseStamped:
        """Generate a PoseStamped message from a 10D action tensor."""

        ee_pos, ee_rot6d, grip = action[:3], action[3:9], action[9]
        ee_quat = self.tf.forward(torch.tensor(ee_rot6d))
        ee_msg = PoseStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.pose.position.x = float(ee_pos[0])
        ee_msg.pose.position.y = float(ee_pos[1])
        ee_msg.pose.position.z = float(ee_pos[2])
        ee_msg.pose.orientation.x = float(ee_quat[1])
        ee_msg.pose.orientation.y = float(ee_quat[2])
        ee_msg.pose.orientation.z = float(ee_quat[3])
        ee_msg.pose.orientation.w = float(ee_quat[0])

        grip_msg = Float32()
        grip_msg.data = float(grip)

        self.get_logger().info(f"Publishing action: Position = {ee_pos.round(4)} | Gripper = {grip:.4f} | Timestamp = {time.monotonic() - self.start_time}")

        return ee_msg, grip_msg

def log_policy(obs, actions, start_time, *, max_channels=6):
    def _shape(x):
        return tuple(x.shape) if hasattr(x, "shape") else f"(len={len(x)})"

    bar = "=" * 80
    print(f"\n{bar}\nCYCLE SUMMARY\n{bar}")
    for k, v in obs.items():
        kind = type(v).__name__
        if k.endswith("_timestamps"):
            print(f"{k:<20} {kind:<14} {_shape(v)}")
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"{k:<20} {kind:<14} {_shape(v)}")
        else:
            print(f"{k:<20} {kind:<14} {v}")

    print(f"\nObservations Passed to Model\n{'-'*80}")

    # Print all observations
    if "pose" in obs and isinstance(obs["pose"], torch.Tensor):
        for ind, curr_obs in enumerate(obs["pose"][0]):
            pose = curr_obs.cpu().numpy()
            pose_timestamp = obs["pose_timestamps"][ind]
            pos, rot6d, grip = pose[:3], pose[3:9], pose[9]
            print(f"Pose {ind + 1}: Position = {pos.round(4)} | Rotation = {rot6d.round(4)} | Gripper = {grip:.4f} | Timestamp = {pose_timestamp}")

    if "zed_rgb" in obs and isinstance(obs["zed_rgb"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["zed_rgb"][0]):
            zed_rgb_timestamp = obs["zed_rgb_timestamps"][ind]
            print(f"Zed RGB Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {zed_rgb_timestamp}")

    if "rs_side_rgb" in obs and isinstance(obs["rs_side_rgb"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["rs_side_rgb"][0]):
            rs_side_rgb_timestamp = obs["rs_side_rgb_timestamps"][ind]
            print(f"RealSense RGB Side Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {rs_side_rgb_timestamp}")

    if "rs_wrist_rgb" in obs and isinstance(obs["rs_wrist_rgb"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["rs_wrist_rgb"][0]):
            rs_wrist_rgb_timestamp = obs["rs_wrist_rgb_timestamps"][ind]
            print(f"RealSense RGB WristImage {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {rs_wrist_rgb_timestamp}")

    if actions is not None:
        print(f"\nActions to be published\n{'-'*80}")
        for ind, curr_act in enumerate(actions):
            print(f"Action {ind + 1}: Position = {curr_act[:3].round(4)} | Rotation = {curr_act[3:9].round(4)} | Gripper = {curr_act[9]:.4f} | Timestamp = {time.monotonic() - start_time}")

    print(bar + "\n")

def crop_frame(img: np.ndarray, crop_size: Tuple[int, int], offset_x: Optional[int] = None, offset_y: Optional[int] = None) -> np.ndarray:
    """Return a cropped view of *img* (HWC) matching the Zarr-dataset logic.

    ``offset_x`` / ``offset_y`` shift the crop centre. Positive moves
    towards bottom-right, negative towards top-left. ``None`` → centre crop.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected (H, W, C), got {img.shape}")

    crop_w, crop_h = crop_size
    H, W, _ = img.shape
    if crop_h > H or crop_w > W:
        raise ValueError(f"Crop {crop_h}x{crop_w} larger than frame {H}x{W}")

    cx, cy = W // 2, H // 2
    if offset_x is not None:
        cx += offset_x
    if offset_y is not None:
        cy += offset_y

    start_x = np.clip(cx - crop_w // 2, 0, W - crop_w)
    start_y = np.clip(cy - crop_h // 2, 0, H - crop_h)
    end_x   = start_x + crop_w
    end_y   = start_y + crop_h

    return img[start_y:end_y, start_x:end_x]

def load_policy(model_path):
    """Load diffusion policy model from checkpoint"""
    # Load checkpoint
    path = pathlib.Path(model_path)
    payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

    # Extract configuration
    cfg = payload['cfg']
    print(cfg)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.policy)

    # Load weights
    model.load_state_dict(payload['state_dicts']['model'])
    
    # Move to device and set evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.reset()
    model.eval()
    model.num_inference_steps = 16 #32  # Number of diffusion steps
    
    # Set up noise scheduler
    noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    model.noise_scheduler = noise_scheduler
    
    return model

def load_shape_meta(model_path: str):
    """
    Return the `shape_meta` dict stored in the checkpoint.
    """
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    return payload["cfg"].policy.shape_meta      # <-- C,H,W per observation key

def load_preprocess_meta(zarr_root: Union[str, Path]) -> dict:
    """Return the preprocessing dictionary stored at the dataset root.

    The Zarr conversion script stores a *single* ``preprocess`` attribute at the
    root level. This helper returns a **plain ``dict``** (so it is JSON-serialisable)
    or an **empty dict** if the attribute does not exist.
    """
    root = zarr.open(str(zarr_root), mode="r")
    return dict(root.attrs.get("preprocess", {}))


def get_cam_preprocess(meta: Mapping[str, Any], cam: str) -> dict:
    """Extract camera-specific preprocessing settings from *meta*.

    The return dict is guaranteed to have the keys ``crop``, ``resize`` and
    ``color_jitter`` (each may be empty). Use it like::

        cam_meta = get_cam_preprocess(meta, "rs_side")
        if cam_meta["crop"]["enabled"]:
            img = crop_frame(img, cam_meta["crop"]["size"],
                            offset_x=cam_meta["crop"]["offset"][0],
                            offset_y=cam_meta["crop"]["offset"][1])
    """
    crop_meta   = meta.get("crop", {}).get(cam, {})
    resize_meta = meta.get("resize", {})
    color_meta  = meta.get("color_jitter", {})

    # Normalise defaults so callers don't need key‑existence guards.
    crop_meta   = {
        "enabled": crop_meta.get("enabled", False),
        "size":    tuple(crop_meta.get("size", (0, 0))),
        "offset":  tuple(crop_meta.get("offset", (None, None))),
    }
    resize_meta = {
        "enabled": resize_meta.get("enabled", False),
        "target":  tuple(resize_meta.get("target", (0, 0))),
    }
    color_meta  = {
        "enabled":    color_meta.get("enabled", False),
        "brightness": color_meta.get("brightness", 0.0),
        "contrast":   color_meta.get("contrast", 0.0),
        "saturation": color_meta.get("saturation", 0.0),
        "hue":        color_meta.get("hue", 0.0),
    }
    return {
        "crop": crop_meta,
        "resize": resize_meta,
        "color_jitter": color_meta,
        }

# Create helper method to save images
def save_data(obs_dict=None, debug_dir=Path("./inference/2d_dp_debug_alex")):
    """Save RGB and depth images for debugging"""
    debug_dir.mkdir(parents=True, exist_ok=True)
    # print(f"Saving data to {debug_dir}")
    
    if obs_dict is None:
        print("No observations to save")
        return
        
    # Save the most recent observation (last in the horizon)
    if "rs_side_rgb" in obs_dict and obs_dict["rs_side_rgb"] is not None:
        # print("Saving RealSense RGB Side image")
        rs_tensor = obs_dict["rs_side_rgb"]
        if isinstance(rs_tensor, torch.Tensor):
            # print(f"rs_tensor shape: {rs_tensor.shape}")
            # Get the most recent frame: [1, T, C, H, W] -> [C, H, W]
            rs_rgb = rs_tensor[0, -1].cpu().numpy()  # Last frame in horizon
            
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            rs_img = (rs_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            # Convert RGB back to BGR for OpenCV
            # rs_img = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(debug_dir/f"rs_side_rgb.jpg"), rs_img)
    
    # Save RealSense RGB Wrist if available
    if "rs_wrist_rgb" in obs_dict and obs_dict["rs_wrist_rgb"] is not None:
        # print("Saving RealSense RGB Wrist image")
        rs_tensor = obs_dict["rs_wrist_rgb"]
        if isinstance(rs_tensor, torch.Tensor):
            # print(f"rs_tensor shape: {rs_tensor.shape}")
            # Get the most recent frame: [1, T, C, H, W] -> [C, H, W]
            rs_rgb = rs_tensor[0, -1].cpu().numpy()  # Last frame in horizon
            
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            rs_img = (rs_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            # Convert RGB back to BGR for OpenCV
            # rs_img = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(debug_dir/f"rs_wrist_rgb.jpg"), rs_img)
    
    # Save ZED RGB if available
    if "zed_rgb" in obs_dict and obs_dict["zed_rgb"] is not None:
        # print("Saving ZED RGB image")
        zed_tensor = obs_dict["zed_rgb"]
        if isinstance(zed_tensor, torch.Tensor):
            # print(f"zed_tensor shape: {zed_tensor.shape}")
            # Get the most recent frame: [1, T, C, H, W] -> [C, H, W]
            zed_rgb = zed_tensor[0, -1].cpu().numpy()  # Last frame in horizon
            
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            zed_img = (zed_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            # Convert RGB back to BGR for OpenCV
            # zed_img = cv2.cvtColor(zed_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(debug_dir/f"zed_rgb.jpg"), zed_img)

def monitor_keys(policy_node):
    """Background thread that listens for *p/u/r* keystrokes using *pygame*.

    * **p** - Pause policy
    * **u** - Unpause / resume
    * **r** - Reset xArm to home pose
    """
    try:
        import pygame
    except ImportError:
        policy_node.get_logger().warning("pygame not installed; keyboard control disabled.")
        return

    pygame.init()
    pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Robot Control")
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    policy_node.pause_policy()
                elif event.key == pygame.K_u:
                    policy_node.resume_policy()
                elif event.key == pygame.K_r:
                    policy_node.reset_xarm()
        clock.tick(60)

def log_inference_timing(start_time, wait_time, prep_time, inference_time, queue_time, total_loop_time):
    """Log the inference timing of the policy"""
    print(f"\n{'='*60}")
    print(f"INFERENCE LOOP TIMING (iteration at {time.monotonic() - start_time:.3f}s)")
    print(f"{'='*60}")
    print(f"Wait for new data:     {wait_time*1000:.2f} ms")
    print(f"Data preparation:      {prep_time*1000:.2f} ms") 
    print(f"Model inference:       {inference_time*1000:.2f} ms")
    print(f"Queue actions:         {queue_time*1000:.2f} ms")
    print(f"TOTAL LOOP TIME:       {total_loop_time*1000:.2f} ms")
    print(f"{'='*60}\n")
    
