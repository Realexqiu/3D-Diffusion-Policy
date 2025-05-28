#!/usr/bin/env python3
"""
Enhanced Policy Node with Point Cloud Generation
===============================================

This script combines the 3D diffusion policy inference with
Grounded SAM-2 for point cloud generation from RGB-D images.
"""

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
import threading
import pathlib
# import pygame
import numpy as np
import torch
from cv_bridge import CvBridge
import cv2
import time
import diffusers
import hydra
import dill
from pathlib import Path
# Multiprocessing imports
import multiprocessing
from multiprocessing import Process, Manager, Queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters

gendp_path = '/home/alex/Documents/DT-Diffusion-Policy/gendp/gendp'

if gendp_path not in sys.path:
    sys.path.append(gendp_path)

from gendp.model.common.rotation_transformer import RotationTransformer


class PolicyNode3D(Node):
    def __init__(self, shared_obs, action_queue, start_time, shape_meta: dict):
        super().__init__('Policy_Node')
        np.set_printoptions(suppress=True, precision=4)

        # --- QoS tuned for high-rate image streams ---
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Subscribers
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, '/robot_pose', qos_profile=sensor_qos)
        self.zed_rgb_sub = message_filters.Subscriber(self, Image, '/zed_image/rgb', qos_profile=sensor_qos)
        self.zed_depth_sub = message_filters.Subscriber(self, Image, '/zed_image/depth', qos_profile=sensor_qos)
        self.rs_color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=sensor_qos)
        self.gripper_sub = message_filters.Subscriber(self, Float32, '/gripper_state', qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.zed_rgb_sub, self.gripper_sub, self.rs_color_sub], 
            queue_size=30,         # how many "unmatched" msgs to keep
            slop=0.3,           
            allow_headerless=True)  # Allow messages without headers
        
        self.sync.registerCallback(self.synced_obs_callback)
        
        # Publishers
        self.gripper_pub = self.create_publisher(Float32, '/gripper_position', 1)
        self.reset_xarm_pub = self.create_publisher(Bool, '/reset_xarm', 1)
        self.pause_xarm_pub = self.create_publisher(Bool, '/pause_xarm', 1)
        self.pub_robot_pose = self.create_publisher(PoseStamped, '/xarm_position', 1)
        
        self._bridge = CvBridge()

        self.start_time = start_time
        self.shape_meta = shape_meta

        # Shared data and action queue
        self.shared_obs = shared_obs
        self.action_queue = action_queue
        self.dt = 0.05  
        
        # Timers
        self.create_timer(self.dt, self.timer_callback)
        self.create_timer(self.dt, self.update_observation)

        # Horizon for keeping recent observations
        self.observation_horizon = 2
        
        # Buffers for observations
        self.pose_buffer = []
        self.zed_rgb_buffer = []
        self.rs_color_buffer = []

        # Rotation transformer
        self.tf = RotationTransformer('rotation_6d', 'quaternion')

        # Control state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paused = False
        self.even = True
        self.pending_actions = []

        # Jiggle gripper to activate it
        msg = Float32()
        msg.data = 0.05
        self.gripper_pub.publish(msg)
        time.sleep(0.5)
        msg = Float32()
        msg.data = 0.0
        self.gripper_pub.publish(msg)
        self.gripper_state = 0.0

        # Reset robot to home position
        self.reset_xarm_pub.publish(Bool(data=True))

        self.get_logger().info("2D Diffusion Policy Node for Baseline Gripper Initialized!")

    # Create helper method to save images
    def save_data(self, zed_rgb_msg, rs_msg=None, debug_dir=Path("2d_dp_debug")):
        """Save RGB and depth images for debugging"""
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Check if zed_rgb_msg is a numpy array
        if isinstance(zed_rgb_msg, np.ndarray):
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            if zed_rgb_msg.shape[0] == 3:  # CHW format
                # Denormalize back to 0-255 range, transpose to HWC, and convert to uint8
                zed_rgb_img = (zed_rgb_msg * 255).astype(np.uint8).transpose(1, 2, 0)
            else:
                zed_rgb_img = zed_rgb_msg
        else:
            # Convert RGB image
            zed_rgb_img = self._bridge.imgmsg_to_cv2(zed_rgb_msg, desired_encoding='bgr8')
        
        # Save converted images
        cv2.imwrite(str(debug_dir/f"zed_rgb.jpg"), zed_rgb_img)
        
        # Save RealSense image if provided
        if rs_msg is not None:
            if isinstance(rs_msg, np.ndarray):
                # Convert from CHW float32 normalized to HWC uint8 for OpenCV
                if rs_msg.shape[0] == 3:  # CHW format
                    # Denormalize back to 0-255 range, transpose to HWC, and convert to uint8
                    rs_img = (rs_msg * 255).astype(np.uint8).transpose(1, 2, 0)
                else:
                    rs_img = rs_msg
            else:
                rs_img = self._bridge.imgmsg_to_cv2(rs_msg, desired_encoding='bgr8')
            cv2.imwrite(str(debug_dir/f"rs_rgb.jpg"), rs_img)

    def reset_xarm(self):
        """Reset robot to home position"""
        self.get_logger().info("Reset xarm.")
        ee_pose = PoseStamped()
        ee_pose.header.stamp = self.get_clock().now().to_msg()
        ee_pose.pose.position.x = float(0.1669)
        ee_pose.pose.position.y = float(0.0019)
        ee_pose.pose.position.z = float(0.2308)
        quats = [0.9999, -0.00995, 0.00507, 0.00785]
        ee_pose.pose.orientation.x = float(quats[0])
        ee_pose.pose.orientation.y = float(quats[1])
        ee_pose.pose.orientation.z = float(quats[2])
        ee_pose.pose.orientation.w = float(quats[3])
        self.pub_robot_pose.publish(ee_pose)

        # Set gripper to open position
        gripper_value = float(0.0)
        msg = Float32()
        msg.data = gripper_value
        self.gripper_pub.publish(msg)

    def pause_policy(self):
        """Pause robot execution"""
        self.get_logger().info("Pause policy.")
        self.pause_xarm_pub.publish(Bool(data=True))
        self.paused = True
        self.shared_obs['paused'] = True          # tell inference
        self.pending_actions.clear()              # drop anything queued
        while not self.action_queue.empty():      # flush queue
            _ = self.action_queue.get_nowait()

    def resume_policy(self):
        """Resume robot execution"""
        self.get_logger().info("Resume policy.")
        self.pause_xarm_pub.publish(Bool(data=False))        
        self.paused = False
        self.shared_obs['paused'] = False         # let inference run

    def cleanup(self):
        """Clean up resources"""
        # pygame.quit()

    def update_observation(self):
        """Consolidate the latest *horizon* observations and push to `shared_obs`."""
        min_len = min(len(self.pose_buffer), len(self.zed_rgb_buffer), len(self.rs_color_buffer))
        if min_len < self.observation_horizon:
            return  # not enough data yet

        # Robot Pose
        pose_slice = self.pose_buffer[-self.observation_horizon:]
        pose_np = np.stack([p[0] for p in pose_slice])           # (T, 10)
        pose_tstamps = np.array([p[1] for p in pose_slice])           # (T,)
        pose_tensor = torch.from_numpy(pose_np).unsqueeze(0)         # (1, T, 10)

        # Zed RGB
        zed_rgb_slice = self.zed_rgb_buffer[-self.observation_horizon:]
        zed_rgb_np = np.stack([r[0] for r in zed_rgb_slice])            # (T, 3, H, W)
        zed_rgb_tstamps = np.array([r[1] for r in zed_rgb_slice])            # (T,)
        zed_rgb_tensor = torch.from_numpy(zed_rgb_np).unsqueeze(0)          # (1, T, 3, H, W)

        # RealSense RGB
        rs_slice = self.rs_color_buffer[-self.observation_horizon:]
        rs_np = np.stack([r[0] for r in rs_slice])             # (T, 3, H, W)
        rs_rgb_tstamps = np.array([r[1] for r in rs_slice])             # (T,)
        rs_rgb_tensor = torch.from_numpy(rs_np).unsqueeze(0)           # (1, T, 3, H, W)

        # self.save_data(zed_rgb_slice[0][0], rs_slice[0][0])

        obs_dict = {
            'pose': pose_tensor,
            'zed_color_images': zed_rgb_tensor,
            'rs_color_images': rs_rgb_tensor,
            'pose_timestamps': pose_tstamps,
            'zed_rgb_timestamps': zed_rgb_tstamps,
            'rs_rgb_timestamps': rs_rgb_tstamps
        }

        # Push to shared memory (IPC)
        self.shared_obs['obs'] = obs_dict


    def timer_callback(self):
        # Pull freshly queued actions into the local list
        while not self.action_queue.empty():
            self.pending_actions.append(self.action_queue.get())

        if not self.pending_actions:
            self.shared_obs["exec_done"] = True
            return
        else:
            self.shared_obs["exec_done"] = False

        # Publish the oldest action
        while self.pending_actions:
            action = self.pending_actions.pop(0)

            # Create pose and gripper messages
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

            # grip_msg = Float32()
            # grip_msg.data = float(grip)

            if not self.paused:
                self.get_logger().info(f"Publishing action: Position = {ee_pos.round(4)} | Gripper = {grip:.4f} | Timestamp = {time.monotonic() - self.start_time}")
                self.pub_robot_pose.publish(ee_msg)
                # self.gripper_pub.publish(grip_msg)

            time.sleep(0.1)
        
        self.shared_obs["exec_done"] = True


    def synced_obs_callback(self, pose_msg, zed_rgb_msg, gripper_msg, rs_msg):
        """Process synchronized observations and generate point cloud"""

        # Update gripper state
        self.gripper_state = gripper_msg.data
        
        # Process pose message
        self.pose_callback(pose_msg)

        # Process zed rgb message
        self.zed_rgb_callback(zed_rgb_msg)

        # Process rs rgb message
        self.rs_rgb_callback(rs_msg)

        # Save data
        # self.save_data(zed_rgb_msg, zed_depth_msg, rs_msg)

        self.update_observation()

    def pose_callback(self, msg):
        """Process robot pose"""
        robot_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] 

        # tf.inverse expects [wxyz]
        robot_ori = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] 
        
        # # Convert to 6D rotation representation
        robot_ori_tensor = torch.tensor(robot_ori, dtype=torch.float32)
        robot_ori_6d = self.tf.inverse(robot_ori_tensor)

        # Combine position, orientation
        robot_pose = np.concatenate([robot_pos, robot_ori_6d.numpy(), self.gripper_state], axis=None)
        self.pose_buffer.append((robot_pose, time.monotonic() - self.start_time))
        if len(self.pose_buffer) > self.observation_horizon:
            self.pose_buffer.pop(0)

    def zed_rgb_callback(self, msg):
        """Process zed rgb message"""
        zed_rgb_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f"Zed RGB Image Shape: {zed_rgb_img.shape}")
        zed_rgb_img = self.resize_for_policy(zed_rgb_img, 'zed_color_images')

        self.zed_rgb_buffer.append((zed_rgb_img, time.monotonic() - self.start_time))
        if len(self.zed_rgb_buffer) > self.observation_horizon:
            self.zed_rgb_buffer.pop(0)

    def rs_rgb_callback(self, msg):
        """Process rs rgb message"""
        rs_rgb_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rs_rgb_img = self.resize_for_policy(rs_rgb_img, 'rs_color_images')

        self.rs_color_buffer.append((rs_rgb_img, time.monotonic() - self.start_time))
        if len(self.rs_color_buffer) > self.observation_horizon:
            self.rs_color_buffer.pop(0)

    def resize_for_policy(self, img: np.ndarray, cam_key: str) -> np.ndarray:
        """
        Resize *and* change layout from HWC-BGR (OpenCV) to CHW-RGB
        according to `shape_meta`.
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


def inference_loop(model_path, shared_obs, action_queue, action_horizon = 4, device = "cuda", start_time = 0):

    model = load_diffusion_policy(model_path)
    print("Inference process started.")

    # ─── Wait until first observation ───────────────────────────────
    while shared_obs.get("obs") is None:
        time.sleep(0.05)

    prev_pose_latest = shared_obs["obs"]["pose_timestamps"][-1]
    prev_zed_latest = shared_obs["obs"]["zed_rgb_timestamps"][-1]
    prev_rs_latest = shared_obs["obs"]["rs_rgb_timestamps"][-1]

    while True:
        # If user paused with the keyboard, spin-wait cheaply
        if shared_obs.get("paused", False):
            time.sleep(0.05)
            continue

        while True:
            obs_now = shared_obs["obs"]

            pose_new = np.min(obs_now["pose_timestamps"]) > prev_pose_latest
            zed_new = np.min(obs_now["zed_rgb_timestamps"]) > prev_zed_latest
            rs_new = np.min(obs_now["rs_rgb_timestamps"]) > prev_rs_latest

            if pose_new and zed_new and rs_new and shared_obs['exec_done']:
                break
            time.sleep(0.001)

        # Save newest observation timestamps
        prev_pose_latest = obs_now["pose_timestamps"][-1]
        prev_zed_latest = obs_now["zed_rgb_timestamps"][-1]
        prev_rs_latest = obs_now["rs_rgb_timestamps"][-1]

        # Grab new observations
        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_now.items()} 

        # Extract pose and point cloud
        model_obs = {k: obs[k] for k in ("pose", "zed_color_images", "rs_color_images")}

        # Predict an action-horizon batch
        with torch.no_grad():
            actions = model.predict_action(model_obs)["action"][0].detach().cpu().numpy()

        q_actions = actions[:action_horizon]          # shape (action_horizon, 10)

        # Print observation
        detailed_debug_summary(obs, q_actions, start_time)

        # Fill the queue
        for act in q_actions:
            action_queue.put(act)

        # Sleep to allow for execution latency
        time.sleep(0.75)


def load_diffusion_policy(model_path):
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
    model.num_inference_steps = 32  # Number of diffusion steps
    
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

# def monitor_keys(policy_node):
#     """Monitor keyboard input for robot control"""
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN: 
#                 if event.key == pygame.K_p:
#                     policy_node.pause_policy()
#                 elif event.key == pygame.K_u:
#                     policy_node.resume_policy()
#                 elif event.key == pygame.K_r:
#                     policy_node.reset_xarm()
#         time.sleep(0.01)

def detailed_debug_summary(obs, actions, start_time, *, max_channels=6):
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
            print(f"Pose {ind + 1}: Position = {pos.round(4)} | Gripper = {grip:.4f} | Timestamp = {pose_timestamp}")

    if "zed_color_images" in obs and isinstance(obs["zed_color_images"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["zed_color_images"][0]):
            zed_rgb_timestamp = obs["zed_rgb_timestamps"][ind]
            print(f"Zed RGB Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {zed_rgb_timestamp}")

    if "rs_color_images" in obs and isinstance(obs["rs_color_images"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["rs_color_images"][0]):
            rs_rgb_timestamp = obs["rs_rgb_timestamps"][ind]
            print(f"RealSense RGB Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {rs_rgb_timestamp}")

    if actions is not None:
        print(f"\nActions to be published\n{'-'*80}")
        for ind, curr_act in enumerate(actions):
            print(f"Action {ind + 1}: Position = {curr_act[:3].round(4)} | Gripper = {curr_act[9]:.4f} | Timestamp = {time.monotonic() - start_time}")

    print(bar + "\n")


def main(args=None):
    np.set_printoptions(suppress=True, precision=4)
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    rclpy.init(args=args)
    
    # # Initialize Pygame for keyboard control
    # pygame.init()
    # pygame.display.set_mode((300, 200))
    # pygame.display.set_caption("Robot Control")
    
    # Create shared memory structures
    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=False, exec_done=True)
    action_queue = Queue()
    start_time = time.monotonic()

    # Model path
    model_path = '/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/models/new_setup_two_cam.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape_meta = load_shape_meta(model_path)

    # Start inference process
    inference_process = Process(
        target=inference_loop, 
        args=(model_path, shared_obs, action_queue, 4, device, start_time)
    )
    inference_process.daemon = True
    inference_process.start()

    # Create ROS node
    node = PolicyNode3D(shared_obs, action_queue, start_time, shape_meta=shape_meta)

    # # Start key monitoring thread
    # key_thread = threading.Thread(target=monitor_keys, args=(node,), daemon=True)
    # key_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up resources
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        inference_process.terminate()


if __name__ == '__main__':
    main()