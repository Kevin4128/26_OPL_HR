import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('opl_human_vision')
    default_engine = os.path.join(package_share, 'models', 'yolo11m.engine')
    image_topic = LaunchConfiguration('image_topic')
    detections_topic = LaunchConfiguration('detections_topic')
    engine_path = LaunchConfiguration('engine_path')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    detector = Node(
        package='opl_human_vision',
        executable='head_cam_object_detector',
        name=LaunchConfiguration('detector_name'),
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'detections_topic': detections_topic,
            'engine_path': engine_path,
            'confidence_threshold': 0.35,
            'furniture_confidence_threshold': 0.55,
            'small_object_confidence_threshold': 0.30,
            'nms_threshold': 0.40,
            'use_sim_time': use_sim_time,
        }],
    )
    visualizer = Node(
        package='opl_human_vision',
        executable='object_visualizer.py',
        name=LaunchConfiguration('visualizer_name'),
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'detections_topic': detections_topic,
            'use_sim_time': use_sim_time,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/head_cam/image'),
        DeclareLaunchArgument('detections_topic', default_value='/head_cam/objects'),
        DeclareLaunchArgument('engine_path', default_value=default_engine),
        DeclareLaunchArgument('detector_name', default_value='head_cam_object_detector'),
        DeclareLaunchArgument('visualizer_name', default_value='head_cam_object_visualizer'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_visualizer', default_value='true'),
        detector,
        visualizer,
    ])
