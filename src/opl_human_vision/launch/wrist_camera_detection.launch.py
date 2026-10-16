from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('opl_human_vision')
    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{package_share}/launch/object_detection.launch.py'),
        launch_arguments={
            'image_topic': LaunchConfiguration('image_topic'),
            'detections_topic': LaunchConfiguration('detections_topic'),
            'engine_path': LaunchConfiguration('engine_path'),
            'detector_name': LaunchConfiguration('detector_name'),
            'visualizer_name': LaunchConfiguration('visualizer_name'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'enable_visualizer': LaunchConfiguration('enable_visualizer'),
        }.items(),
    )
    fusion = Node(
        package='opl_human_vision',
        executable='wrist_object_fusion.py',
        name=LaunchConfiguration('fusion_name'),
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'detections_topic': LaunchConfiguration('detections_topic'),
            'objects_topic': LaunchConfiguration('objects_topic'),
            'target_frame': LaunchConfiguration('target_frame'),
            'max_sync_slop': ParameterValue(LaunchConfiguration('max_sync_slop'), value_type=float),
            'max_depth_m': ParameterValue(LaunchConfiguration('max_depth_m'), value_type=float),
            'publish_annotated_image': ParameterValue(
                LaunchConfiguration('publish_annotated_image'), value_type=bool),
            'publish_markers': ParameterValue(LaunchConfiguration('publish_markers'), value_type=bool),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/wrist_cam/image'),
        DeclareLaunchArgument('depth_topic', default_value='/wrist_cam/depth_image'),
        DeclareLaunchArgument('camera_info_topic', default_value='/wrist_cam/camera_info'),
        DeclareLaunchArgument('detections_topic', default_value='/wrist_cam/objects_raw'),
        DeclareLaunchArgument('objects_topic', default_value='/wrist_cam/objects'),
        DeclareLaunchArgument('engine_path', default_value=f'{package_share}/models/yolo11m.engine'),
        DeclareLaunchArgument('detector_name', default_value='wrist_cam_object_detector'),
        DeclareLaunchArgument('visualizer_name', default_value='wrist_cam_object_visualizer'),
        DeclareLaunchArgument('fusion_name', default_value='wrist_object_fusion'),
        DeclareLaunchArgument('target_frame', default_value='base_footprint'),
        DeclareLaunchArgument('max_sync_slop', default_value='0.03'),
        DeclareLaunchArgument('max_depth_m', default_value='5.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_visualizer', default_value='false'),
        DeclareLaunchArgument('publish_annotated_image', default_value='false'),
        DeclareLaunchArgument('publish_markers', default_value='false'),
        detector_launch,
        fusion,
    ])
