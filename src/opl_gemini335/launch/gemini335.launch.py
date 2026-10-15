import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_width', default_value='640', description='Colour and registered depth width'),
        DeclareLaunchArgument('image_height', default_value='480', description='Colour and registered depth height'),
        DeclareLaunchArgument('framerate', default_value='30', description='Colour and depth frames per second'),
        DeclareLaunchArgument('serial_number', default_value='', description='Optional Gemini serial number'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('orbbec_camera'), 'launch', 'gemini_330_series.launch.py')),
            launch_arguments={
                'camera_name': 'camera',
                'serial_number': LaunchConfiguration('serial_number'),
                'color_width': LaunchConfiguration('image_width'),
                'color_height': LaunchConfiguration('image_height'),
                'depth_width': LaunchConfiguration('image_width'),
                'depth_height': LaunchConfiguration('image_height'),
                'color_fps': LaunchConfiguration('framerate'),
                'depth_fps': LaunchConfiguration('framerate'),
                'enable_color': 'true',
                'enable_depth': 'true',
                'depth_registration': 'true',
                'enable_frame_sync': 'true',
                'enable_depth_scale': 'true',
                'enable_point_cloud': 'false',
                'enable_colored_point_cloud': 'false',
                'enable_left_ir': 'false',
                'enable_right_ir': 'false',
                'color_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
            }.items(),
        ),
    ])
