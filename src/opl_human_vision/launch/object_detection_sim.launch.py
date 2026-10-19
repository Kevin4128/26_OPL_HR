"""Object-only perception; start the existing Gazebo robot separately."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    share = get_package_share_directory('opl_human_vision')
    defaults = {
        'model_path': os.path.join(share, 'laptop_models', 'yolo11m.engine'),
        'device_id': '0',
        'image_topic': '/head_cam/image',
        'depth_topic': '/head_cam/depth_image',
        'camera_info_topic': '/head_cam/camera_info',
        'detections_topic': '/detected_objects',
        'fixed_frame': 'odom',
        'optical_frame': 'head_cam_optical_link',
        'inference_hz': '15.0',
        'use_sim_time': 'true',
        'tf_timeout_s': '0.1',
        'tf_retry_period_s': '0.005',
        'enable_visualizer': 'false',
        'annotated_image_topic': '/object_detection/annotated_image',
        'visualizer_hz': '7.5',
    }
    parameters = {
        key: ParameterValue(LaunchConfiguration(key), value_type=(
            int if key == 'device_id' else float if key in ('inference_hz', 'tf_timeout_s', 'tf_retry_period_s', 'visualizer_hz') else
            bool if key == 'use_sim_time' else str))
        for key in defaults if key not in ('enable_visualizer', 'annotated_image_topic', 'visualizer_hz')
    }
    node = LifecycleNode(
        package='opl_human_vision', executable='object_detection_node_exe',
        name='object_detection_node', namespace='', output='screen',
        parameters=[os.path.join(share, 'config', 'object_detection_sim.yaml'), parameters])
    configure = RegisterEventHandler(OnProcessStart(
        target_action=node, on_start=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(node), transition_id=Transition.TRANSITION_CONFIGURE))]))
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node, start_state='configuring', goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(node), transition_id=Transition.TRANSITION_ACTIVATE))]))
    visualizer = Node(
        package='opl_human_vision', executable='object_visualizer.py', output='screen',
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        parameters=[{key: ParameterValue(LaunchConfiguration(key), value_type=(
            float if key == 'visualizer_hz' else bool if key == 'use_sim_time' else str))
            for key in ('image_topic', 'detections_topic', 'annotated_image_topic', 'visualizer_hz', 'use_sim_time')}])
    viewer = Node(
        package='rqt_image_view', executable='rqt_image_view', name='object_image_view',
        arguments=[LaunchConfiguration('annotated_image_topic')], output='screen',
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('enable_visualizer'), "'.lower() == 'true' and ",
            repr(bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))])))
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        configure, activate, node, visualizer, viewer])
