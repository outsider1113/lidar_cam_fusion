from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_cam_fusion',
            executable='fusion_node.py',
            name='fusion_node',
            output='screen',
        ),
    ])