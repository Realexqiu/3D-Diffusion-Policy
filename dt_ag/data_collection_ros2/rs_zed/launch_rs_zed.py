#!/usr/bin/env python3

import sys
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    script_1 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/rs_zed/3d_rs_zed_hdf5_collector.py"
    script_2 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/xarm_spacemouse_ros2.py"
    script_3 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/publish_zed.py"

    # Get the ZED wrapper package path
    # zed_wrapper_dir = get_package_share_directory('zed_wrapper')
    # zed_launch_file = os.path.join(zed_wrapper_dir, 'launch', 'zed_camera.launch.py')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='xarm_rs_zed_collector',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='xarm_spacemouse_ros2',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_3],
            name='publish_zed',
            output='screen'
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',  # Changed name to avoid conflict
            output='screen'
        ),
        # # Include ZED camera launch file
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(zed_launch_file),
        #     launch_arguments={
        #         'camera_model': 'zed2i',  # Change this to your camera model (zed, zed2, zed2i, zedm, zedx, etc.)
        #         'camera_name': 'zed_camera'  # Optional: set a specific camera name
        #     }.items()
        # ),
    ])

def main(argv=sys.argv[1:]):
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())