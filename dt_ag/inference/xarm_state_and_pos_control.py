#!/usr/bin/env python3
"""
Combined XArm State Publisher and Position Controller

This script combines two nodes:
1. XArmStatePublisher: Publishes the current state of the XArm (pose, joint states, gripper)
2. XArmPositionNode: Subscribes to commanded positions and controls the XArm

Running this single script replaces the need for two separate node processes.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R


class CombinedXArmNode(Node):
    """
    A combined ROS node that both publishes XArm state and controls the XArm position.
    """
    def __init__(self):
        super().__init__('combined_xarm_node')
        
        # Declare parameter for xarm IP address
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.213').value
        self.get_logger().info(f'Using xArm with IP: {self.ip}')

        # Initialize XArm
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Current gripper command value
        self.gripper = 0.0
        self.prev_pose = None
        self.is_paused = False
        self.is_resetting = False

        # Subscribers (from XArmPositionNode)
        self.position_sub = self.create_subscription(PoseStamped, 'xarm_position', self.position_callback, 1)
        self.gripper_cmd_sub = self.create_subscription(Float32,'gripper_position', self.gripper_callback, 1)
        self.reset_sub = self.create_subscription(Bool, '/reset_xarm', self.reset_callback, 10)
        self.pause_sub = self.create_subscription(Bool, '/pause_xarm', self.pause_callback, 10)

        # Publishers (from XArmStatePublisher)
        self.pose_publisher = self.create_publisher(PoseStamped, 'robot_pose', 1)
        self.prev_publisher = self.create_publisher(PoseStamped, 'prev_robot_pose', 1)
        self.joints_publisher = self.create_publisher(JointState, 'xarm_joint_states', 1)
        self.gripper_state_pub = self.create_publisher(Float32, 'gripper_state', 10)

        # Timers
        self.state_timer = self.create_timer(1.0 / 30.0, self.state_timer_callback)

        # Reset and gripper open
        self.arm.set_gripper_mode(0)      # location mode
        self.arm.set_gripper_enable(True) # power the driver
        self.arm.set_gripper_speed(5000)  # any speed you like (1-5000)
        self.arm.clean_gripper_error()    # clear residual errors
        self.gripper_callback(Float32(data=0.0))
        self.reset_callback(Bool(data=True))

        self.get_logger().info('Combined XArm node initialized successfully')

    def setup_xarm(self):
        """
        Initialize the XArm with appropriate settings.
        """
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(state=0)  # Ready state
        time.sleep(1)
        self.get_logger().info('XArm initialized successfully')

    def state_timer_callback(self):
        """
        Timer callback to publish current XArm state (pose, joint states, gripper).
        """
        self.get_logger().debug('Publishing XArm state...')

        # Publish previous pose if available
        if self.prev_pose is not None:
            self.publish_previous_pose(self.prev_pose)

        # Get current arm position and joint angles
        code, pose = self.arm.get_position(is_radian=True)
        code, angles = self.arm.get_joint_states(is_radian=True)
        
        if code != 0:
            self.get_logger().warn(f'Failed to get XArm position: error code {code}')
            return

        # Convert position from mm to m
        pose[0:3] = [x / 1000 for x in pose[0:3]]
        
        # Publish current pose and joint angles
        self.publish_pose(pose)
        self.publish_angles(angles[0] if angles else [])
        
        # Store current pose for next iteration
        self.prev_pose = pose

        # Get and publish gripper state
        code, gripper_val = self.arm.get_gripper_position()
        # self.get_logger().info(f"Gripper value: {gripper_val}")
        if code == 0:
            # Normalize gripper value from [0-850] to [0-1] range
            normalized_gripper_val = (gripper_val - 850) / -850.0
            
            msg = Float32()
            msg.data = normalized_gripper_val
            self.gripper_state_pub.publish(msg)

    def position_callback(self, pose_msg: PoseStamped):
        """
        Callback for receiving commanded positions.
        """
        if self.is_paused or self.is_resetting:
            return

        # Extract position (convert from m to mm)
        x_mm = pose_msg.pose.position.x * 1000.0
        y_mm = pose_msg.pose.position.y * 1000.0
        z_mm = pose_msg.pose.position.z * 1000.0

        # Extract quaternion orientation
        qw = pose_msg.pose.orientation.w
        qx = pose_msg.pose.orientation.x
        qy = pose_msg.pose.orientation.y
        qz = pose_msg.pose.orientation.z

        # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
        ar = R.from_quat([qx, qy, qz, qw])
        roll_rad, pitch_rad, yaw_rad = ar.as_euler('xyz', degrees=False)

        # Set the position and orientation
        speed = 30  # mm/s
        self.arm.set_position(
            x=x_mm, y=y_mm, z=z_mm, 
            roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad, 
            speed=speed, is_radian=True, wait=False
        )

    def gripper_callback(self, gripper_msg: Float32):
        """
        Callback for receiving gripper position commands.
        """
        if self.is_paused or self.is_resetting:
            return
            
        # Set gripper position
        self.gripper = gripper_msg.data
        
        # Convert normalized value [0-1] to XArm gripper value [0-850]
        grasp = 850 - 850 * self.gripper
        self.arm.set_gripper_position(grasp, wait=False)

    def reset_callback(self, msg: Bool):
        """
        Reset the XArm to a predefined position.
        """
        if msg.data:
            self.get_logger().info('Resetting XArm position...')
            self.is_resetting = True
            
            # Reset position (in mm and degrees)
            self.arm.set_position(
                x=166.9, y=1.9, z=230.8, 
                roll=179.1, pitch=0, yaw=1.2, 
                speed=100, is_radian=False, wait=True
            )
            
            # Reset gripper
            self.arm.set_gripper_position(850, wait=True)  # Fully open
            
            self.is_resetting = False
            self.get_logger().info('XArm reset complete')

    def pause_callback(self, msg: Bool):
        """
        Pause or resume XArm motion.
        """
        self.is_paused = msg.data
        if self.is_paused:
            self.get_logger().info('XArm motion paused')
        else:
            self.get_logger().info('XArm motion resumed')

    def publish_previous_pose(self, pose):
        """
        Publish the previous XArm pose.
        """
        x, y, z, roll, pitch, yaw = pose

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = x
        msg.pose.position.y = y 
        msg.pose.position.z = z 

        # Convert Euler to quaternion
        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.prev_publisher.publish(msg)

    def publish_angles(self, angles):
        """
        Publish the current XArm joint angles.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = angles
        self.joints_publisher.publish(msg)

    def publish_pose(self, pose):
        """
        Publish the current XArm pose.
        """
        x, y, z, roll, pitch, yaw = pose

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Convert Euler to quaternion
        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(msg)

    def destroy_node(self):
        """
        Cleanly disconnect from XArm before shutting down.
        """
        self.get_logger().info('Disconnecting from XArm...')
        self.arm.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = CombinedXArmNode()
    
    # Use MultiThreadedExecutor for better performance with multiple callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info('Running combined XArm node...')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()