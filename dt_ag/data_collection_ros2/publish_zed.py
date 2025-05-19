#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyzed.sl as sl
import cv2
import numpy as np
from cv_bridge import CvBridge

class ZedImagePublisher(Node):
    def __init__(self):
        super().__init__('zed_image_publisher')

        # Create publishers for both RGB and depth images
        self.rgb_publisher = self.create_publisher(Image, 'zed_image/rgb', 10)
        self.depth_publisher = self.create_publisher(Image, 'zed_image/depth', 10)

        self.bridge = CvBridge()
        # Timer for grabbing and publishing images (4 Hz)
        self.timer = self.create_timer(0.25, self.timer_callback)

        # ==========================================================
        # (A) Initialize the ZED camera with provided settings
        # ==========================================================
        self.zed_cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD2K
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_stabilization = 1
        init_params.camera_image_flip = sl.FLIP_MODE.AUTO
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        status = self.zed_cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"ZED open failed: {status}")
            # Ensure the node is properly destroyed on failure
            self.destroy_node()
            raise RuntimeError(f"ZED open failed: {status}")

        self.get_logger().info("ZED camera initialized.")

        # ==========================================================
        # (B) Retrieve and log camera calibration information
        # ==========================================================
        camera_information = self.zed_cam.get_camera_information()
        calibration_params = camera_information.camera_configuration.calibration_parameters

        # Access intrinsic parameters for the left camera
        left_cam_intrinsics = calibration_params.left_cam
        focal_left_x = left_cam_intrinsics.fx
        focal_left_y = left_cam_intrinsics.fy
        principal_point_left_x = left_cam_intrinsics.cx
        principal_point_left_y = left_cam_intrinsics.cy
        distortion_coeffs_left = left_cam_intrinsics.disto # Radial and tangential distortion coefficients

        # Access stereo parameters
        baseline_translation_x = calibration_params.stereo_transform.get_translation().get()[0] # Translation between left and right eye on x-axis (baseline)
        # Note: ZED SDK provides baseline directly as tx

        # Log the calibration information
        self.get_logger().info("--- Camera Calibration Parameters (Left Camera) ---")
        self.get_logger().info(f"Focal Length (fx): {focal_left_x}")
        self.get_logger().info(f"Focal Length (fy): {focal_left_y}")
        self.get_logger().info(f"Principal Point (cx): {principal_point_left_x}")
        self.get_logger().info(f"Principal Point (cy): {principal_point_left_y}")
        self.get_logger().info(f"Distortion Coefficients (k1, k2, p1, p2, k3): {distortion_coeffs_left}")
        self.get_logger().info(f"Stereo Baseline (tx): {baseline_translation_x} meters")
        self.get_logger().info("-------------------------------------------------")


        self.zed_left_image = sl.Mat()
        self.zed_depth_map = sl.Mat()
        # Define the desired resolution for retrieval
        self.desired_res = sl.Resolution(2208, 1242) # HD2K resolution

    def timer_callback(self):
        # Grab a new frame from the ZED camera
        if self.zed_cam.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image in BGRA format
            self.zed_cam.retrieve_image(self.zed_left_image, sl.VIEW.LEFT, resolution=self.desired_res)
            frame_bgra = self.zed_left_image.get_data()
            # Convert from BGRA to BGR for compatibility with cv_bridge
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # Retrieve the depth map
            self.zed_cam.retrieve_measure(self.zed_depth_map, sl.MEASURE.DEPTH, resolution=self.desired_res)
            depth_data = self.zed_depth_map.get_data()

            try:
                # Convert OpenCV images to ROS Image messages
                # Use bgr8 encoding for color image
                rgb_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                # Use 32FC1 encoding for depth map (float 32-bit, 1 channel)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_data, encoding="32FC1")

                # Set headers with current time and frame_id
                current_time = self.get_clock().now().to_msg()

                rgb_msg.header.stamp = current_time
                rgb_msg.header.frame_id = "zed_camera" # Consistent frame ID

                depth_msg.header.stamp = current_time
                depth_msg.header.frame_id = "zed_camera" # Consistent frame ID

                # Publish messages
                self.rgb_publisher.publish(rgb_msg)
                self.depth_publisher.publish(depth_msg)

                self.get_logger().debug("Published ZED RGB and depth images.")

            except Exception as e:
                self.get_logger().error(f"Image conversion or publishing error: {e}")
                # Continue to next timer callback even if one fails

    def destroy_node(self):
        # Close the ZED camera connection when the node is destroyed
        if self.zed_cam.is_opened():
            self.zed_cam.close()
            self.get_logger().info("ZED camera closed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ZedImagePublisher()
    try:
        # Keep the node alive
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        # Ensure the node is destroyed and resources are released
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
