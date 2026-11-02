"""Backward-compatible entry point; on the Pi_5 branch this is pi5_hailo.launch.py.

All arguments are forwarded, e.g.
    ros2 launch opl_human_vision test_detector.launch.py camera_type:=gemini335 enable_visualizer:=true
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('opl_human_vision'), 'launch', 'pi5_hailo.launch.py']))),
    ])
