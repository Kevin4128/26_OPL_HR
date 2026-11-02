"""Human detection / tracking / recognition on Raspberry Pi 5 + Hailo-8L.

Everything that touches camera frames runs in ONE multi-threaded component
container with intra-process communication:

    Orbbec Gemini 335 driver -> detector (yolov8s_pose) -> tracker (+ re-ID)
                             -> recognizer (SCRFD + ArcFace + re-ID)

so the 640x480 frames are passed as shared pointers instead of being
serialised over DDS to every node, and all Hailo models share one VDevice
that the HailoRT scheduler time-slices (pose gets the highest priority).

    ros2 launch opl_human_vision pi5_hailo.launch.py enable_visualizer:=true

Use camera_type:=realsense for a RealSense, camera_type:=none to bring your
own camera (publishing camera/color/image_raw + camera/depth/image_raw), or
camera_type:=image test_image:=/path/to/people.jpg to replay a still image
inside the container (no depth) - handy for load tests without a camera.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def _models_dir():
    return os.path.join(get_package_share_directory('opl_human_vision'), 'models', 'hailo8l')


# name, default, description
ARGS = [
    # Camera
    ('camera_type', 'gemini335', 'gemini335 (driver loaded in the vision container), realsense, image, or none'),
    ('test_image', '/usr/local/hailo/resources/images/zidane.jpg', 'Still image replayed when camera_type:=image'),
    ('image_width', '640', 'Colour / registered depth width'),
    ('image_height', '480', 'Colour / registered depth height'),
    ('framerate', '30', 'Camera fps; the detector drops frames it cannot keep up with'),
    ('serial_number', '', 'Optional Gemini serial number'),
    # Measured on the Pi 5: MJPG ~89% container CPU vs YUYV ~98% (driver YUYV->RGB costs more than libjpeg-turbo).
    ('gemini_color_format', 'MJPG', 'Colour stream format: MJPG (lowest Pi CPU) or YUYV'),
    ('gemini_align_mode', 'HW', 'Depth-to-colour alignment: HW (on the camera, no Pi CPU) or SW'),
    ('enable_visualizer', 'false', 'Show the OpenCV visualisation window'),
    ('container_threads', '4', 'Executor threads in the vision container (Pi 5 has 4 cores)'),
    ('timing_debug', 'false', 'Log per-stage wall times ([timing] lines on the vision_timing logger)'),

    # Hailo
    ('hailo_multi_process', 'false',
     'Share the Hailo-8L with other processes through hailort.service (needed if e.g. another Hailo app runs)'),

    # Models (HEFs from scripts/fetch_hailo_models.sh)
    ('detector_model_path', 'yolov8s_pose.hef', 'YOLOv8-pose HEF'),
    ('scrfd_model_path', 'scrfd_2.5g.hef', 'SCRFD face detector HEF'),
    ('face_model_path', 'arcface_mobilefacenet.hef', 'Face embedder HEF: arcface_mobilefacenet.hef (fast) or arcface_r50.hef (accurate, ~9x slower)'),
    ('reid_model_path', 'repvgg_a0_person_reid_512.hef', 'Person re-ID HEF'),

    # Detector
    ('detector_confidence_threshold', '0.55', 'Person score threshold'),
    ('detector_nms_threshold', '0.65', 'Pose-aware NMS IoU threshold'),
    # Measured with 2 people in view: uncapped ~22 Hz but recognition steps 2x slower;
    # 20 Hz keeps tracking smooth and leaves Hailo time for SCRFD/ArcFace/re-ID.
    ('detector_max_hz', '20', 'Cap detector rate (0 = as fast as the Hailo allows; 15 favours recognition)'),

    # Tracker
    ('tracker_max_age', '45', 'Max missed updates to keep a lost track'),
    ('tracker_max_publish_age', '1', 'Max missed updates before hiding an unmatched track'),
    ('tracker_min_hits', '3', 'Detections needed to confirm a track'),
    ('tracker_iou_threshold', '0.30', 'IoU gate for association (lower suits the lower Pi frame rate)'),
    ('appearance_reid_enabled', 'true', 'Re-ID assisted recovery after occlusion'),
    # RepVGG re-ID cosine values are lower than OSNet's; calibrate with tools in README.
    ('appearance_match_threshold', '0.70', 'Min cosine similarity for appearance recovery (RepVGG-A0)'),
    ('appearance_reid_max_age', '45', 'Max missed updates eligible for appearance recovery'),
    ('appearance_refresh_interval', '15', 'Clean matches between gallery refreshes'),
    ('appearance_spatial_base_ratio', '0.10', 'Base recovery radius as a fraction of the image size'),
    ('appearance_spatial_speed_ratio_per_sec', '0.75', 'Recovery radius growth per second'),

    # Recognizer
    ('scrfd_conf_threshold', '0.50', 'SCRFD face score threshold'),
    ('scrfd_nms_threshold', '0.40', 'SCRFD NMS threshold'),
    # ArcFace cosine similarities sit lower than AdaFace's; calibrate per README.
    ('face_match_threshold', '0.45', 'Face cosine similarity threshold (ArcFace)'),
    ('body_match_threshold', '0.70', 'Body cosine similarity threshold (RepVGG-A0)'),
    ('min_confirm_frames', '15', 'Face embeddings required to enrol'),
    ('min_confirm_duration_sec', '2.0', 'Minimum enrolment duration'),
    ('identity_lock_grace_sec', '3.0', 'Keep a raw-track to identity lock during short occlusions'),
    ('identity_validation_interval_sec', '0.5', 'Seconds between deep re-validation of locked identities'),
    ('session_reid_timeout_sec', '5.0', 'Lifetime of anonymous identities after disappearance'),
    ('reid_spatial_gate_max_age_sec', '2.0', 'Max age of a 3D position used for gating'),
    ('reid_spatial_gate_base_m', '0.75', 'Base allowed 3D displacement'),
    ('reid_spatial_gate_max_speed_mps', '2.5', 'Max plausible walking speed'),
    ('reid_depth_gate_base_m', '0.75', 'Base allowed depth change'),
    ('landmarks_on_cached_frames', 'false', 'Also run SCRFD on cached-identity frames (only refreshes drawn landmarks)'),
]


def _model(context, name):
    value = LaunchConfiguration(name).perform(context)
    return value if os.path.isabs(value) else os.path.join(_models_dir(), value)


def _vision(context):
    cfg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    camera_type = cfg('camera_type')
    if camera_type == 'realsense':
        image_topic = 'camera/camera/color/image_raw'
        depth_topic = 'camera/camera/aligned_depth_to_color/image_raw'
    else:
        image_topic = 'camera/color/image_raw'
        depth_topic = 'camera/depth/image_raw'

    multi_process = cfg('hailo_multi_process').lower() == 'true'
    intra = [{'use_intra_process_comms': True}]

    for name in ('detector_model_path', 'scrfd_model_path', 'face_model_path', 'reid_model_path'):
        if not os.path.isfile(_model(context, name)):
            raise RuntimeError(
                f"{name}: '{_model(context, name)}' not found. Run "
                "src/opl_human_vision/scripts/fetch_hailo_models.sh and rebuild, "
                "or pass an absolute HEF path.")

    components = []
    if camera_type == 'image':
        components.append(ComposableNode(
            package='image_publisher',
            plugin='image_publisher::ImagePublisher',
            name='test_image_publisher',
            parameters=[{
                'filename': cfg('test_image'),
                'publish_rate': float(cfg('framerate')),
                'frame_id': 'camera_color_optical_frame',
            }],
            remappings=[('image_raw', image_topic), ('camera_info', 'camera/color/camera_info')],
            extra_arguments=intra,
        ))
    elif camera_type == 'gemini335':
        width, height, fps = int(cfg('image_width')), int(cfg('image_height')), int(cfg('framerate'))
        components.append(ComposableNode(
            package='orbbec_camera',
            plugin='orbbec_camera::OBCameraNodeDriver',
            name='camera',
            namespace='camera',
            parameters=[{
                'camera_name': 'camera',
                'serial_number': cfg('serial_number'),
                'enable_color': True,
                'enable_depth': True,
                'enable_left_ir': False,
                'enable_right_ir': False,
                'enable_point_cloud': False,
                'enable_colored_point_cloud': False,
                'color_width': width, 'color_height': height, 'color_fps': fps,
                'color_format': cfg('gemini_color_format'),
                'depth_width': width, 'depth_height': height, 'depth_fps': fps,
                'depth_registration': True,
                'align_mode': cfg('gemini_align_mode'),
                'enable_frame_sync': True,
                'enable_depth_scale': True,
                # Vendor "low CPU" profile: hardware instead of software noise filter.
                'enable_noise_removal_filter': False,
                'enable_hardware_noise_removal_filter': True,
                'color_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
                # Only raw transport: avoids compressed/theora publishers per stream.
                'color.image_raw.enable_pub_plugins': ['image_transport/raw'],
                'depth.image_raw.enable_pub_plugins': ['image_transport/raw'],
            }],
            extra_arguments=intra,
        ))

    components += [
        ComposableNode(
            package='opl_human_vision',
            plugin='opl_human_vision::HumanDetectorNode',
            name='human_detector_node',
            parameters=[{
                'image_topic': image_topic,
                'detections_topic': 'raw_detections',
                'model_path': _model(context, 'detector_model_path'),
                'confidence_threshold': float(cfg('detector_confidence_threshold')),
                'nms_threshold': float(cfg('detector_nms_threshold')),
                'max_inference_hz': float(cfg('detector_max_hz')),
                'hailo_multi_process': multi_process,
                'hailo_priority': 24,
            }],
            extra_arguments=intra,
        ),
        ComposableNode(
            package='opl_human_vision',
            plugin='opl_human_vision::HumanTrackerNode',
            name='human_tracker_node',
            parameters=[{
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'detections_topic': 'raw_detections',
                'tracked_topic': 'tracked_humans',
                'max_age': int(cfg('tracker_max_age')),
                'max_publish_age': int(cfg('tracker_max_publish_age')),
                'min_hits': int(cfg('tracker_min_hits')),
                'iou_threshold': float(cfg('tracker_iou_threshold')),
                'reid_model_path': _model(context, 'reid_model_path'),
                'appearance_reid_enabled': cfg('appearance_reid_enabled').lower() == 'true',
                'appearance_match_threshold': float(cfg('appearance_match_threshold')),
                'appearance_reid_max_age': int(cfg('appearance_reid_max_age')),
                'appearance_refresh_interval': int(cfg('appearance_refresh_interval')),
                'appearance_spatial_base_ratio': float(cfg('appearance_spatial_base_ratio')),
                'appearance_spatial_speed_ratio_per_sec': float(cfg('appearance_spatial_speed_ratio_per_sec')),
                'hailo_multi_process': multi_process,
                'hailo_priority': 16,
            }],
            extra_arguments=intra,
        ),
        ComposableNode(
            package='opl_human_vision',
            plugin='opl_human_vision::FaceRecognizerNode',
            name='face_recognizer_node',
            parameters=[{
                'image_topic': image_topic,
                'tracked_topic': 'tracked_humans',
                'recognized_topic': 'recognized_humans',
                'scrfd_model_path': _model(context, 'scrfd_model_path'),
                'face_model_path': _model(context, 'face_model_path'),
                'reid_model_path': _model(context, 'reid_model_path'),
                'scrfd_conf_threshold': float(cfg('scrfd_conf_threshold')),
                'scrfd_nms_threshold': float(cfg('scrfd_nms_threshold')),
                'match_threshold': float(cfg('face_match_threshold')),
                'body_match_threshold': float(cfg('body_match_threshold')),
                'min_confirm_frames': int(cfg('min_confirm_frames')),
                'min_confirm_duration_sec': float(cfg('min_confirm_duration_sec')),
                'identity_lock_grace_sec': float(cfg('identity_lock_grace_sec')),
                'identity_validation_interval_sec': float(cfg('identity_validation_interval_sec')),
                'session_reid_timeout_sec': float(cfg('session_reid_timeout_sec')),
                'reid_spatial_gate_max_age_sec': float(cfg('reid_spatial_gate_max_age_sec')),
                'reid_spatial_gate_base_m': float(cfg('reid_spatial_gate_base_m')),
                'reid_spatial_gate_max_speed_mps': float(cfg('reid_spatial_gate_max_speed_mps')),
                'reid_depth_gate_base_m': float(cfg('reid_depth_gate_base_m')),
                'landmarks_on_cached_frames': cfg('landmarks_on_cached_frames').lower() == 'true',
                'hailo_multi_process': multi_process,
                'hailo_priority': 12,
            }],
            extra_arguments=intra,
        ),
    ]

    container = ComposableNodeContainer(
        name='vision_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        parameters=[{'thread_num': int(cfg('container_threads'))}],
        arguments=(['--ros-args', '--log-level', 'vision_timing:=debug']
                   if cfg('timing_debug').lower() == 'true' else []),
        composable_node_descriptions=components,
        output='screen',
    )

    visualizer = Node(
        package='opl_human_vision',
        executable='visualizer.py',
        name='pipeline_visualizer',
        parameters=[{'image_topic': image_topic}],
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        output='screen',
    )

    actions = [container, visualizer]
    if camera_type == 'realsense':
        profile = f"{cfg('image_width')}x{cfg('image_height')}x{cfg('framerate')}"
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
            launch_arguments={
                'align_depth.enable': 'true',
                'depth_module.depth_profile': profile,
                'rgb_camera.color_profile': profile,
            }.items(),
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        # HailoRT writes ./hailort.log by default; keep it out of the source tree.
        SetEnvironmentVariable('HAILORT_LOGGER_PATH', os.path.expanduser('~/.ros/log')),
        *[DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in ARGS],
        OpaqueFunction(function=_vision),
        Node(
            package='opl_human_vision',
            executable='lifecycle_activator.py',
            name='lifecycle_activator',
            output='screen',
        ),
    ])
