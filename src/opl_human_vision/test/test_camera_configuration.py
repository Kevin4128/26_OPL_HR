"""Resolve both camera configurations without starting drivers or model inference."""

import runpy
import sys
from unittest.mock import patch

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import ComposableNodeContainer
from launch_ros.utilities import evaluate_parameters


def check_camera(path, camera, prefix, depth_topic):
    components = []

    def capture_container(**kwargs):
        components.extend(kwargs['composable_node_descriptions'])
        return ComposableNodeContainer(**kwargs)

    with patch('launch_ros.actions.ComposableNodeContainer', capture_container):
        description = runpy.run_path(path)['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations['camera_type'] = camera
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    parameters = {}
    for component in components:
        plugin = perform_substitutions(context, component.node_plugin)
        parameters[plugin] = evaluate_parameters(context, component.parameters)[0]
    detector = parameters['opl_human_vision::HumanDetectorNode']
    tracker = parameters['opl_human_vision::HumanTrackerNode']
    recognizer = parameters['opl_human_vision::FaceRecognizerNode']
    assert detector['image_topic'] == tracker['image_topic'] == recognizer['image_topic']
    assert tracker['image_topic'] == prefix + '/color/image_raw'
    assert tracker['depth_topic'] == depth_topic
    assert tracker['camera_info_topic'] == prefix + '/color/camera_info'
    assert tracker['fixed_frame'] == 'odom'
    assert detector['confidence_threshold'] == tracker['high_confidence_threshold'] == recognizer['minimum_learning_confidence'] == 0.65
    assert detector['recovery_confidence_threshold'] == 0.30
    assert tracker['prediction_hold_sec'] == 0.5
    assert tracker['depth_registered_to_rgb'] is True
    assert tracker['depth_slop_sec'] == 0.02
    assert tracker['registered_frame_aliases'] == ('',)
    assert tracker['observation_wait_sec'] == recognizer['observation_wait_sec'] == 0.1
    assert tracker['observation_buffer_size'] == recognizer['observation_buffer_size'] == 60
    context.launch_configurations['depth_topic'] = '/test/no_depth'
    context.launch_configurations['registered_frame_aliases'] = "['known_registered_frame']"
    tracker_component = next(c for c in components if
                             perform_substitutions(context, c.node_plugin).endswith('HumanTrackerNode'))
    overridden = evaluate_parameters(context, tracker_component.parameters)[0]
    assert overridden['depth_topic'] == '/test/no_depth'
    assert overridden['registered_frame_aliases'] == ('known_registered_frame',)


if __name__ == '__main__':
    check_camera(sys.argv[1], 'gemini335', 'camera', 'camera/depth/image_raw')
    check_camera(sys.argv[1], 'realsense', 'camera/camera',
                 'camera/camera/aligned_depth_to_color/image_raw')
    print('Gemini and RealSense topics, calibration and parameter types passed')
