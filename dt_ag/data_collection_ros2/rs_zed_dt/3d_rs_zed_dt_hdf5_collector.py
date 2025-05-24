#!/usr/bin/env python3

import os
import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# ROS messages
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

# For time synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2  # only used for color conversion from BGRA to BGR

## Plug in Procedure: Realsense first, then ZED, then DenseTact Left (tape), then DenseTact Right (no tape)

class XArmDataCollection(Node):
    def __init__(self):
        super().__init__('xarm_data_collection_node')
        self.get_logger().info("Initializing data_collection_node with approximate sync.")

        # ==========================================================
        # Create message_filters Subscribers for RealSense + Robot
        # ==========================================================
        self.pose_sub = Subscriber(self, PoseStamped, 'robot_position_action')
        self.gripper_sub = Subscriber(self, Float32, 'gripper_position')
        self.rs_color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.rs_depth_sub = Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.zed_color_sub = Subscriber(self, Image, 'zed_image/rgb')
        self.zed_depth_sub = Subscriber(self, Image, 'zed_image/depth')
        self.dt_left_sub = Subscriber(self, Image, 'RunCamera/image_raw_8')
        self.dt_right_sub = Subscriber(self, Image, 'RunCamera/image_raw_10')

        # ApproximateTimeSynchronizer for all five
        self.sync = ApproximateTimeSynchronizer(
            [
                self.pose_sub,
                self.gripper_sub,
                self.rs_color_sub,
                self.rs_depth_sub,
                self.dt_left_sub,
                self.dt_right_sub,
                self.zed_color_sub,
                self.zed_depth_sub
            ],
            queue_size=30,
            slop=0.1,
            allow_headerless=True  # needed because Float32 doesn't have a header
        )
        self.sync.registerCallback(self.synced_callback)

        # ==========================================================
        # Start/End Demo as separate subscriptions
        # ==========================================================

        self.start_sub = self.create_subscription(Bool, 'start_demo', self.start_demo_callback, 10)
        self.end_sub = self.create_subscription(Bool, 'end_demo', self.end_demo_callback, 10)

        # ==========================================================
        # State and storage
        # ==========================================================
        self.is_collecting = False

        # start number
        self.demo_count = 0

        # Robot data
        self.pose_data = []
        self.gripper_data = []

        # RealSense
        self.rs_color_frames = []
        self.rs_depth_frames = []

        # ZED
        self.zed_color_frames = []
        self.zed_depth_frames = []

        # DenseTacts
        self.dt_left_frames = []
        self.dt_right_frames = []

        # Throttle to 5 Hz
        self.collection_frequency = 5.0
        self.min_period = 1.0 / self.collection_frequency
        self.last_saved_time = self.get_clock().now() - Duration(seconds=self.min_period)

        # Episode timing
        self.episode_start_time = None

    ####################################################################
    # Start/End Demos
    ####################################################################
    def start_demo_callback(self, msg: Bool):
        if msg.data and not self.is_collecting:
            self.get_logger().info("Starting a new demonstration.")
            self.is_collecting = True
            self.episode_start_time = self.get_clock().now()

            # Clear buffers
            self.pose_data.clear()
            self.gripper_data.clear()
            self.rs_color_frames.clear()
            self.rs_depth_frames.clear()
            self.zed_color_frames.clear()
            self.zed_depth_frames.clear()
            self.dt_left_frames.clear()
            self.dt_right_frames.clear()

    def end_demo_callback(self, msg: Bool):
        if msg.data and self.is_collecting:
            self.get_logger().info("Ending demonstration and saving.")
            self.is_collecting = False
            self.save_demonstration()

            end_time = self.get_clock().now()
            duration = (end_time - self.episode_start_time).nanoseconds / 1e9 if self.episode_start_time else 0.0

            # Compute number of frames saved
            num_frames = len(self.pose_data)
            rate = num_frames / duration if duration > 0 else float('nan')

            # Log summary
            self.get_logger().info(f"Episode {self.demo_count} ended: Length = {duration:.2f}s, Num_frames = {num_frames}, Freq. = {rate:.2f}Hz")
            self.demo_count += 1

    ####################################################################
    # Time sychronization callback
    ####################################################################
    def synced_callback(self,
                        pose_msg: PoseStamped,
                        grip_msg: Float32,
                        rs_color_msg: Image,
                        rs_depth_msg: Image,
                        dt_left_msg: Image,
                        dt_right_msg: Image,
                        zed_color_msg: Image,
                        zed_depth_msg: Image):
        """
        Called when (Pose, Gripper, RealSense color, RealSense depth)
        arrive (approximately) at the same time. We'll also grab from the ZED here.
        """        
        # Skip if not collecting
        if not self.is_collecting:
            return
        
        now = self.get_clock().now()
        elapsed = (now - self.last_saved_time).nanoseconds / 1e9
        if elapsed < self.min_period:
            return
        
        self.get_logger().info(f"In synced_callback")

        self.last_saved_time = now

        # Robot pose/gripper
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        
        # Transformer for training expects w,x,y,z order for quaternion
        self.pose_data.append([p.x, p.y, p.z, o.w, o.x, o.y, o.z])

        self.gripper_data.append(grip_msg.data)

        # Parse RealSense color
        rs_color_np_bgr = self.parse_color_image(rs_color_msg)  # shape (H, W, 3)
        rs_color_np_rgb = cv2.cvtColor(rs_color_np_bgr, cv2.COLOR_BGR2RGB)
        self.rs_color_frames.append(rs_color_np_rgb)

        # Parse RealSense depth
        rs_depth_np = self.parse_depth_image(rs_depth_msg)  # shape (H, W)
        self.rs_depth_frames.append(rs_depth_np)

        # Parse ZED color
        zed_np = self.parse_color_image(zed_color_msg)  # shape (H, W, 3)
        self.zed_color_frames.append(zed_np)

        # Parse ZED depth
        zed_depth_np = self.parse_depth_image(zed_depth_msg)  # shape (H, W)
        self.zed_depth_frames.append(zed_depth_np)

        # Capture from DenseTacts cameras
        dt_left_np = self.parse_dt_image(dt_left_msg)  # shape (H, W, 3)
        dt_right_np = self.parse_dt_image(dt_right_msg)  # shape (H, W, 3)
        self.dt_left_frames.append(dt_left_np)
        self.dt_right_frames.append(dt_right_np)

    ####################################################################
    # Save the demonstration to HDF5
    ####################################################################

    def save_demonstration(self):
        # Convert lists to numpy arrays
        pose_array = np.array(self.pose_data, dtype=np.float32)
        
        # convert mm to meters for pose
        pose_array[:, :3] /= 1000.0

        grip_array = np.array(self.gripper_data, dtype=np.float32)

        # Create last_pose by shifting pose_array by one index and replacing the new first pose with the original first pose
        last_pose_array = np.roll(pose_array, shift=-1, axis=0)
        last_pose_array[0] = pose_array[0]

        # RealSense data
        rs_color_stack = np.stack(self.rs_color_frames, axis=0) if len(self.rs_color_frames) > 0 else []
        rs_depth_stack = np.stack(self.rs_depth_frames, axis=0) if len(self.rs_depth_frames) > 0 else []

        # ZED data
        zed_color_stack = np.stack(self.zed_color_frames, axis=0) if len(self.zed_color_frames) > 0 else []
        zed_depth_stack = np.stack(self.zed_depth_frames, axis=0) if len(self.zed_depth_frames) > 0 else []

        # DenseTacts data
        dt_left_stack = np.stack(self.dt_left_frames, axis=0) if len(self.dt_left_frames) > 0 else []
        dt_right_stack = np.stack(self.dt_right_frames, axis=0) if len(self.dt_right_frames) > 0 else []

        # Construct filename
        save_dir = os.path.join(os.getcwd(), "demo_data")
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"episode_{self.demo_count}.hdf5")

        with h5py.File(filename, "w") as f:
            # Robot data
            f.create_dataset("pose", data=pose_array)
            f.create_dataset("gripper", data=grip_array)
            f.create_dataset("last_pose", data=last_pose_array)

            # RealSense color
            if len(rs_color_stack) > 0:
                f.create_dataset("rs_color_images", data=rs_color_stack, compression="lzf")

            # RealSense depth
            if len(rs_depth_stack) > 0:
                f.create_dataset("rs_depth_images", data=rs_depth_stack, compression="lzf")

            # ZED color
            if len(zed_color_stack) > 0:
                f.create_dataset("zed_color_images", data=zed_color_stack, compression="lzf")

            # ZED depth
            if len(zed_depth_stack) > 0:
                f.create_dataset("zed_depth_images", data=zed_depth_stack, compression="lzf")

            # DenseTacts color
            if len(dt_left_stack) > 0:
                f.create_dataset("dt_left_images", data=dt_left_stack, compression="lzf")
            if len(dt_right_stack) > 0:
                f.create_dataset("dt_right_images", data=dt_right_stack, compression="lzf")

        self.get_logger().info(f"Saved demonstration to {filename}")

    ####################################################################
    # Helper Functions
    ####################################################################  
    def parse_dt_image(self, img_msg: Image) -> np.ndarray:
        """
        Parse DenseTact camera images, which publish in RGB but we want them in BGR.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step
        data = img_msg.data

        # Convert raw bytes to 1D array of type uint8 and reshape
        np_data = np.frombuffer(data, dtype=np.uint8)
        np_data_2d = np_data.reshape((height, step))

        # Assume 3 channels: step == width * 3 (no row padding)
        channels = 3
        expected_bytes_per_row = width * channels
        np_data_2d_sliced = np_data_2d[:, :expected_bytes_per_row]

        # Reshape to (H, W, 3)
        color_img = np_data_2d_sliced.reshape((height, width, channels))

        # If the DT images are published in "rgb8", swap channels to get BGR
        if img_msg.encoding == "rgb8":
            color_img = color_img[..., ::-1]

        return color_img

    def parse_color_image(self, img_msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image (e.g., 'bgr8' or 'rgb8') into a NumPy array (H,W,3).
        Adjust for your specific encoding as needed.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step  # bytes per row
        data = img_msg.data  # raw bytes

        # Convert to a 1D numpy array of dtype uint8
        np_data = np.frombuffer(data, dtype=np.uint8)
        
        # Reshape to (height, step)
        np_data_2d = np_data.reshape((height, step))

        # For color images, we expect step == width * 3 (if no row padding)
        channels = 3
        expected_bytes_per_row = width * channels
        np_data_2d_sliced = np_data_2d[:, :expected_bytes_per_row]

        # Finally reshape to (H, W, C)
        color_img = np_data_2d_sliced.reshape((height, width, channels))
        return color_img
    
    def parse_depth_image(self, img_msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image depth into a NumPy array (H, W).
        Handles both 32FC1 and 16UC1 encodings.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step  # bytes per row
        data = img_msg.data
        
        if img_msg.encoding == "32FC1":
            # Each pixel is a 32-bit float => 4 bytes
            floats_per_row = step // 4
            np_data = np.frombuffer(data, dtype=np.float32)
            
            # Reshape to (height, floats_per_row)
            depth_2d = np_data.reshape((height, floats_per_row))
            
            # Slice out the valid columns
            depth_2d_sliced = depth_2d[:, :width]
            
        elif img_msg.encoding == "16UC1":
            # Each pixel is a 16-bit unsigned int => 2 bytes
            ints_per_row = step // 2
            np_data = np.frombuffer(data, dtype=np.uint16)
            
            # Reshape to (height, ints_per_row)
            depth_2d = np_data.reshape((height, ints_per_row))
            
            # Slice out the valid columns
            depth_2d_sliced = depth_2d[:, :width]
            
            # Convert uint16 millimeters to float32 meters
            depth_2d_sliced = depth_2d_sliced.astype(np.float32) / 1000.0
        else:
            self.get_logger().error(f"Unsupported depth encoding: {img_msg.encoding}")
            depth_2d_sliced = np.zeros((height, width), dtype=np.float32)
            
        return depth_2d_sliced
    
    def destroy_node(self) -> bool:
        if hasattr(self, 'zed_cam') and self.zed_cam.is_opened():
            self.zed_cam.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmDataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()