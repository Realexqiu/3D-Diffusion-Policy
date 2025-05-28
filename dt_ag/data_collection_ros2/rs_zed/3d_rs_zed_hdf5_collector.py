#!/usr/bin/env python3

import os
import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

from message_filters import Subscriber, ApproximateTimeSynchronizer

import pyzed.sl as sl
import cv2  # for color conversion


class XArmDataCollection(Node):
    def __init__(self):
        super().__init__('xarm_data_collection_node')
        self.get_logger().info("Initializing data_collection_node with approximate sync + 5 Hz throttle.")

        # Subscribers for robot and cameras
        self.pose_sub = Subscriber(self, PoseStamped, 'robot_position_action')
        self.gripper_sub = Subscriber(self, Float32, 'gripper_position')
        self.rs_color_sub = Subscriber(self, Image, '/camera/realsense_camera/color/image_raw')
        self.rs_depth_sub = Subscriber(self, Image, '/camera/realsense_camera/depth/image_raw')
        self.zed_color_sub = Subscriber(self, Image, 'zed_image/rgb')
        self.zed_depth_sub = Subscriber(self, Image, 'zed_image/depth')

        # ApproximateTimeSynchronizer
        self.sync = ApproximateTimeSynchronizer(
            [
                self.pose_sub,
                self.gripper_sub,
                self.rs_color_sub,
                self.rs_depth_sub,
                self.zed_color_sub,
                self.zed_depth_sub
            ],
            queue_size=30,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_callback)

        # Start/End demo
        self.start_sub = self.create_subscription(Bool, 'start_demo', self.start_demo_callback, 10)
        self.end_sub = self.create_subscription(Bool, 'end_demo',   self.end_demo_callback,   10)

        # State
        self.is_collecting = False
        self.demo_count = 0

        self.pose_data = []
        self.gripper_data = []
        self.rs_color_frames = []
        self.rs_depth_frames = []
        self.zed_color_frames = []
        self.zed_depth_frames = []

        # Throttle to 5 Hz
        self.collection_frequency = 5.0
        self.min_period = 1.0 / self.collection_frequency
        self.last_saved_time = self.get_clock().now() - Duration(seconds=self.min_period)

        # Episode timing
        self.episode_start_time = None

    def start_demo_callback(self, msg: Bool):
        if msg.data and not self.is_collecting:
            self.get_logger().info("Starting a new demonstration.")
            self.is_collecting = True

            # Clear buffers
            self.pose_data.clear()
            self.gripper_data.clear()
            self.rs_color_frames.clear()
            self.rs_depth_frames.clear()
            self.zed_color_frames.clear()
            self.zed_depth_frames.clear()

            # Reset throttle
            self.last_saved_time = self.get_clock().now() - Duration(seconds=self.min_period)
            # Start timing
            self.episode_start_time = self.get_clock().now()

    def end_demo_callback(self, msg: Bool):
        if msg.data and self.is_collecting:
            self.get_logger().info(f"Ending episode {self.demo_count}")
            end_time = self.get_clock().now()
            duration = (end_time - self.episode_start_time).nanoseconds / 1e9 if self.episode_start_time else 0.0

            # Save data
            self.is_collecting = False
            self.save_demonstration()

            # Compute number of frames saved
            num_frames = len(self.pose_data)
            rate = num_frames / duration if duration > 0 else float('nan')

            # Log summary
            self.get_logger().info(f"Episode {self.demo_count} ended: Length = {duration:.2f}s, Num_frames = {num_frames}, Freq. = {rate:.2f}Hz")

            self.demo_count += 1

    def synced_callback(self, pose_msg: PoseStamped, grip_msg: Float32, rs_color_msg: Image, rs_depth_msg: Image, zed_color_msg: Image, zed_depth_msg: Image):
        if not self.is_collecting:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_saved_time).nanoseconds / 1e9
        if elapsed < self.min_period:
            return

        self.last_saved_time = now

        # Robot pose & gripper
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        self.pose_data.append([p.x, p.y, p.z, o.w, o.x, o.y, o.z])
        self.gripper_data.append(grip_msg.data)

        # RealSense color
        rs_color_np_bgr = self.parse_color_image(rs_color_msg)
        rs_color_np_rgb = cv2.cvtColor(rs_color_np_bgr, cv2.COLOR_BGR2RGB)
        self.rs_color_frames.append(rs_color_np_rgb)

        # RealSense depth
        rs_depth_np = self.parse_depth_image(rs_depth_msg)
        self.rs_depth_frames.append(rs_depth_np)

        # ZED color
        zed_np = self.parse_color_image(zed_color_msg)
        self.get_logger().info(f"Zed RGB Image Shape: {zed_np.shape}")
        self.zed_color_frames.append(zed_np)

        # ZED depth
        zed_depth_np = self.parse_depth_image(zed_depth_msg)
        self.zed_depth_frames.append(zed_depth_np)

        self.get_logger().info(f"Collected {len(self.pose_data)} frames")

    def save_demonstration(self):
        pose_array = np.array(self.pose_data, dtype=np.float32)
        pose_array[:, :3] /= 1000.0
        grip_array = np.array(self.gripper_data, dtype=np.float32)

        last_pose_array = np.roll(pose_array, shift=-1, axis=0)
        last_pose_array[0] = pose_array[0]

        rs_color_stack = np.stack(self.rs_color_frames,  axis=0) if self.rs_color_frames else []
        rs_depth_stack = np.stack(self.rs_depth_frames,  axis=0) if self.rs_depth_frames else []
        zed_color_stack = np.stack(self.zed_color_frames, axis=0) if self.zed_color_frames else []
        zed_depth_stack = np.stack(self.zed_depth_frames, axis=0) if self.zed_depth_frames else []

        save_dir = os.path.join(os.getcwd(), "demo_data")
        os.makedirs(save_dir, exist_ok=True)
        fn = os.path.join(save_dir, f"episode_{self.demo_count}.hdf5")
        with h5py.File(fn, "w") as f:
            f.create_dataset("pose", data=pose_array)
            f.create_dataset("gripper", data=grip_array)
            f.create_dataset("last_pose", data=last_pose_array)
            if len(rs_color_stack):  f.create_dataset("rs_color_images", data=rs_color_stack,  compression="lzf")
            if len(rs_depth_stack):  f.create_dataset("rs_depth_images", data=rs_depth_stack,  compression="lzf")
            if len(zed_color_stack): f.create_dataset("zed_color_images", data=zed_color_stack, compression="lzf")
            if len(zed_depth_stack): f.create_dataset("zed_depth_images", data=zed_depth_stack, compression="lzf")

        self.get_logger().info(f"Saved demonstration to {fn}")

    def parse_color_image(self, img_msg: Image) -> np.ndarray:
        h, w, step = img_msg.height, img_msg.width, img_msg.step
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape((h, step))
        return arr[:, : w * 3].reshape((h, w, 3))

    def parse_depth_image(self, img_msg: Image) -> np.ndarray:
        h, w, step = img_msg.height, img_msg.width, img_msg.step
        if img_msg.encoding == "32FC1":
            arr = np.frombuffer(img_msg.data, dtype=np.float32).reshape((h, step // 4))
            return arr[:, :w]
        elif img_msg.encoding == "16UC1":
            arr = np.frombuffer(img_msg.data, dtype=np.uint16).reshape((h, step // 2))
            return arr[:, :w].astype(np.float32) / 1000.0
        else:
            self.get_logger().error(f"Unsupported depth encoding: {img_msg.encoding}")
            return np.zeros((h, w), dtype=np.float32)

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmDataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
