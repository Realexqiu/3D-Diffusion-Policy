#!/usr/bin/env python3

import numpy as np
import time
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped
import argparse
import os


class RobotCommander(Node):
    """
    Node to read saved pose/action data and send commands to the robot.
    """
    
    def __init__(self):
        super().__init__('robot_commander')
        
        # Publishers
        self.pub_robot_pose = self.create_publisher(PoseStamped, '/xarm_position', 1)
        self.gripper_pub = self.create_publisher(Float32, '/gripper_position', 1)
        self.reset_xarm_pub = self.create_publisher(Bool, '/reset_xarm', 1)
        self.pause_xarm_pub = self.create_publisher(Bool, '/pause_xarm', 1)
        
        # Initialize 6D rotation transformer (assuming you have this)
        try:
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            self.tf = RotationTransformer('rotation_6d', 'quaternion')
        except ImportError:
            self.get_logger().warn("Could not import RotationTransformer, using identity transform")
            self.tf = None
        
        self.start_time = time.monotonic()
        
        self.get_logger().info("Robot Commander initialized")

    def generate_ee_action_msg(self, action: np.ndarray):
        """Generate PoseStamped and Float32 messages from a 10D action array."""
        
        ee_pos = action[:3]
        ee_rot6d = action[3:9] 
        grip = action[9]
        
        # Convert 6D rotation to quaternion
        if self.tf is not None:
            try:
                ee_quat = self.tf.forward(torch.tensor(ee_rot6d, dtype=torch.float32))
                ee_quat = ee_quat.numpy()
            except Exception as e:
                self.get_logger().warn(f"Rotation conversion failed: {e}")
                self.get_logger().warn("Using identity quaternion as fallback")
                ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        else:
            # Fallback: use identity quaternion
            ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Create pose message
        ee_msg = PoseStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = "base_link"
        ee_msg.pose.position.x = float(ee_pos[0])
        ee_msg.pose.position.y = float(ee_pos[1])
        ee_msg.pose.position.z = float(ee_pos[2])
        ee_msg.pose.orientation.w = float(ee_quat[0])  # w first in quaternion
        ee_msg.pose.orientation.x = float(ee_quat[1])
        ee_msg.pose.orientation.y = float(ee_quat[2])
        ee_msg.pose.orientation.z = float(ee_quat[3])
        
        # Create gripper message
        grip_msg = Float32()
        grip_msg.data = float(grip)
        
        return ee_msg, grip_msg

    def execute_trajectory(self, pose_data: np.ndarray, action_data: np.ndarray, 
                          rate_hz: float = 5.0, use_actions: bool = True):
        """
        Execute a trajectory by publishing either pose or action data.
        
        Args:
            pose_data: Array of poses (T, 10)
            action_data: Array of actions (T, 10) 
            rate_hz: Publishing rate in Hz
            use_actions: If True, use action_data; if False, use pose_data
        """
        
        data_to_use = action_data if use_actions else pose_data
        data_name = "actions" if use_actions else "poses"
        
        self.get_logger().info(f"Executing trajectory with {len(data_to_use)} {data_name}")
        self.get_logger().info(f"Publishing at {rate_hz} Hz")
        
        sleep_time = 1.0 / rate_hz
        
        for i, data_point in enumerate(data_to_use):
            ee_msg, grip_msg = self.generate_ee_action_msg(data_point)
            
            # Publish messages
            self.pub_robot_pose.publish(ee_msg)
            self.gripper_pub.publish(grip_msg)
            
            # Log progress
            if i % 10 == 0 or i == len(data_to_use) - 1:
                pos = data_point[:3]
                grip = data_point[9]
                elapsed = time.monotonic() - self.start_time
                self.get_logger().info(
                    f"Step {i+1}/{len(data_to_use)}: "
                    f"Pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                    f"Grip={grip:.3f} Time={elapsed:.2f}s"
                )
            
            time.sleep(sleep_time)
        
        self.get_logger().info("Trajectory execution completed")

    def reset_robot(self):
        """Send reset command to robot."""
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_xarm_pub.publish(reset_msg)
        self.get_logger().info("Reset command sent")

    def pause_robot(self, pause: bool = True):
        """Send pause/unpause command to robot."""
        pause_msg = Bool()
        pause_msg.data = pause
        self.pause_xarm_pub.publish(pause_msg)
        action = "Paused" if pause else "Unpaused"
        self.get_logger().info(f"Robot {action}")


def load_trajectory_data(pose_file: str, action_file: str):
    """Load pose and action data from .npy files."""
    
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    if not os.path.exists(action_file):
        raise FileNotFoundError(f"Action file not found: {action_file}")
    
    pose_data = np.load(pose_file)
    action_data = np.load(action_file)
    
    print(f"Loaded pose data shape: {pose_data.shape}")
    print(f"Loaded action data shape: {action_data.shape}")
    
    return pose_data, action_data


def main():
    parser = argparse.ArgumentParser(description='Send saved trajectory to robot')
    parser.add_argument('--pose_file', type=str, default='debug_images/pose.npy',
                       help='Path to pose .npy file')
    parser.add_argument('--action_file', type=str, default='debug_images/action.npy', 
                       help='Path to action .npy file')
    parser.add_argument('--rate', type=float, default=20.0,
                       help='Publishing rate in Hz')
    parser.add_argument('--use_actions', action='store_true', default=True,
                       help='Use action data instead of pose data')
    parser.add_argument('--use_poses', dest='use_actions', action='store_false',
                       help='Use pose data instead of action data')
    parser.add_argument('--reset_first', action='store_true',
                       help='Send reset command before starting')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Load trajectory data
        pose_data, action_data = load_trajectory_data(args.pose_file, args.action_file)
        
        # Create robot commander
        commander = RobotCommander()
        
        # Optional reset
        if args.reset_first:
            commander.reset_robot()
            time.sleep(2.0)  # Wait for reset
        
        # Execute trajectory
        commander.execute_trajectory(
            pose_data, action_data, 
            rate_hz=args.rate, 
            use_actions=args.use_actions
        )
        
    except KeyboardInterrupt:
        print("\nTrajectory execution interrupted by user")
    except Exception as e:
        # print(f"Error: {e}")
        raise e
    finally:
        if 'commander' in locals():
            commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 