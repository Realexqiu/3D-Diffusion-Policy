#!/usr/bin/env python3

import sys
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    script_1 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/publish_zed.py"
    script_2 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/xarm_state_and_pos_control.py"
    script_3 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/publish_dt.py"
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='publish_zed',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='xarm_state_and_pos_control',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_3],
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
