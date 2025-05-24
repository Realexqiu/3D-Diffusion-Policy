#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import subprocess

class DualCameraPublisher(Node):
    def __init__(self):
        super().__init__('dt_dual_camera_publisher')
        
        # Frame counter for delayed camera settings
        self.count = 0
        self.settings_applied = False
        
        # Hardcoded camera IDs
        self.left_camera_id = 8  # DenseTact Left (with tape)
        self.right_camera_id = 10  # DenseTact Right (no tape)
        
        # Target settings values
        self.target_wb_temp = 6500
        self.target_exposure = 50
        
        # Create publishers for both cameras
        self.left_publisher = self.create_publisher(Image, f"RunCamera/image_raw_{self.left_camera_id}", 20)
        
        self.right_publisher = self.create_publisher(Image, f"RunCamera/image_raw_{self.right_camera_id}", 20)
        
        # Create camera capture objects
        self.left_cap = cv2.VideoCapture(self.left_camera_id)
        self.right_cap = cv2.VideoCapture(self.right_camera_id)
        
        # Set the resolution for both cameras
        for cap in [self.left_cap, self.right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
        
        # Check if cameras opened successfully
        if not self.left_cap.isOpened():
            self.get_logger().error(f"Unable to open left camera with id {self.left_camera_id}")
        else:
            self.get_logger().info(f"Left camera (id {self.left_camera_id}) opened successfully")
            
        if not self.right_cap.isOpened():
            self.get_logger().error(f"Unable to open right camera with id {self.right_camera_id}")
        else:
            self.get_logger().info(f"Right camera (id {self.right_camera_id}) opened successfully")
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Create timer to publish camera frames
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz publishing rate

    def get_camera_setting(self, video_id, control_name):
        """Get the current value of a camera control.
        
        Args:
            video_id (int): The video device ID (/dev/video{video_id})
            control_name (str): The name of the control to check
            
        Returns:
            int: The current value of the control, or None if not found
        """
        try:
            # Run the v4l2-ctl command to get the current value
            cmd = f"v4l2-ctl --device /dev/video{video_id} -C {control_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                # Extract value from output like "white_balance_temperature: 6500"
                value_str = result.stdout.strip().split(': ')[1]
                try:
                    return int(value_str)
                except ValueError:
                    # Handle cases where value isn't an integer
                    return value_str
            else:
                self.get_logger().warning(f"Failed to get {control_name} for camera {video_id}: {result.stderr}")
                return None
        except Exception as e:
            self.get_logger().error(f"Error getting camera setting: {e}")
            return None
        
    def set_manual_wb(self, video_id, wb_temp=6500):
        """Set manual white balance for a camera, but only if needed."""
        # First check current white balance setting
        current_wb_auto = self.get_camera_setting(video_id, "white_balance_automatic")
        current_wb_temp = self.get_camera_setting(video_id, "white_balance_temperature")
        
        self.get_logger().info(f"Camera {video_id} current WB auto: {current_wb_auto}, WB temp: {current_wb_temp}")
        
        # Only change settings if needed
        if current_wb_auto != 0 or current_wb_temp != wb_temp:
            self.get_logger().info(f"Setting WB for camera {video_id} to {wb_temp}")
            commands = [
                f"v4l2-ctl --device /dev/video{video_id} -c white_balance_automatic=0",
                f"v4l2-ctl --device /dev/video{video_id} -c white_balance_automatic=1",
                f"v4l2-ctl --device /dev/video{video_id} -c white_balance_automatic=0",
                f"v4l2-ctl --device /dev/video{video_id} -c white_balance_temperature={wb_temp}"
            ]
            for cmd in commands:
                ret = os.system(cmd)
                if ret != 0:
                    self.get_logger().warning(f"Command failed: {cmd}")
                else:
                    self.get_logger().info(f"Successfully executed: {cmd}")
        else:
            self.get_logger().info(f"Camera {video_id} WB already set correctly")

    def set_manual_exposure(self, video_id, exposure_time):
        """Set manual exposure for a camera, but only if needed."""
        # First check current exposure settings
        current_auto_exposure = self.get_camera_setting(video_id, "auto_exposure")
        current_exposure_time = self.get_camera_setting(video_id, "exposure_time_absolute")
        
        self.get_logger().info(f"Camera {video_id} current auto exposure: {current_auto_exposure}, exposure time: {current_exposure_time}")
        
        # Only change settings if needed
        if current_auto_exposure != 1 or current_exposure_time != exposure_time:
            self.get_logger().info(f"Setting exposure for camera {video_id} to {exposure_time}")
            commands = [
                f"v4l2-ctl --device /dev/video{video_id} -c auto_exposure=3",
                f"v4l2-ctl --device /dev/video{video_id} -c auto_exposure=1",
                f"v4l2-ctl --device /dev/video{video_id} -c exposure_time_absolute={exposure_time}"
            ]
            for cmd in commands:
                ret = os.system(cmd)
                if ret != 0:
                    self.get_logger().warning(f"Command failed: {cmd}")
                else:
                    self.get_logger().info(f"Successfully executed: {cmd}")
        else:
            self.get_logger().info(f"Camera {video_id} exposure already set correctly")

    def apply_camera_settings(self):
        """Apply white balance first, then exposure settings, but only if needed"""
        self.get_logger().info("Checking and applying camera settings after 30 frames")
        
        # First set white balance for both cameras
        self.set_manual_wb(self.left_camera_id, self.target_wb_temp)
        self.set_manual_wb(self.right_camera_id, self.target_wb_temp)
        
        # Then set exposure for both cameras
        self.set_manual_exposure(self.left_camera_id, self.target_exposure)
        self.set_manual_exposure(self.right_camera_id, self.target_exposure)
        
        self.settings_applied = True
        self.get_logger().info("Camera settings check completed")

    def publish_frame(self, cap, publisher, camera_id):
        """
        Capture and publish a frame from the specified camera.
        
        Args:
            cap: OpenCV VideoCapture object
            publisher: ROS publisher for this camera
            camera_id: ID of the camera (for logging)
        """
        ret, frame = cap.read()
        
        if not ret:
            self.get_logger().warning(f"Failed to capture image from camera {camera_id}")
            return

        # Force conversion to numpy array
        frame = np.array(frame)

        # Resize to 320 x 180
        frame = cv2.resize(frame, (320, 180))

        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"camera_{camera_id}_frame"
        
        # Publish the message
        publisher.publish(msg)

    def timer_callback(self):
        """
        Timer callback to capture and publish frames from both cameras.
        """
        # Increment counter
        self.count += 1
        
        # Check if we should apply camera settings
        if self.count == 30 and not self.settings_applied:
            self.apply_camera_settings()
        
        # Publish frames from both cameras
        self.publish_frame(self.left_cap, self.left_publisher, self.left_camera_id)
        self.publish_frame(self.right_cap, self.right_publisher, self.right_camera_id)
        
    def destroy_node(self):
        """
        Clean up resources when the node is destroyed.
        """
        # Release camera resources
        if self.left_cap.isOpened():
            self.left_cap.release()
        
        if self.right_cap.isOpened():
            self.right_cap.release()
            
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = DualCameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()