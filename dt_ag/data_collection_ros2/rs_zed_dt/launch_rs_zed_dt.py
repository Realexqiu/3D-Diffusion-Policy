#!/usr/bin/env python3

import sys
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    script_1 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/rs_zed_dt/3d_rs_zed_dt_hdf5_collector.py"
    script_2 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/xarm_spacemouse_ros2.py"
    script_3 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/publish_zed.py"
    script_4 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/rs_zed_dt/publish_dt.py"

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='xarm_rs_zed_dt_collector',
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
        ExecuteProcess(
            cmd=['python3', script_4],
            name='publish_dt',
            output='screen'
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            output='screen'
        )
    ])

def main(argv=sys.argv[1:]):
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())
