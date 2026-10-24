import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg_share = get_package_share_directory('opl_human_vision')
    models_dir = os.path.join(pkg_share, 'laptop_models')

    # Default model file paths
    default_detector_model = os.path.join(models_dir, 'yolov8n-pose.engine')
    default_scrfd_model    = os.path.join(models_dir, 'scrfd_2.5g.engine')
    default_adaface_model  = os.path.join(models_dir, 'adaface_ir50.engine')
    default_osnet_model    = os.path.join(models_dir, 'osnet_x1_0.engine')

    # -------------------------------------------------------------------------
    # 1. Camera & Hardware Arguments
    # -------------------------------------------------------------------------
    width_arg   = DeclareLaunchArgument('image_width', default_value='640', description='Camera frame width')
    height_arg  = DeclareLaunchArgument('image_height', default_value='480', description='Camera frame height')
    fps_arg     = DeclareLaunchArgument('framerate', default_value='30', description='Camera fps')
    device_arg  = DeclareLaunchArgument('video_device', default_value='0', description='Video device ID')
    camera_type_arg = DeclareLaunchArgument('camera_type', default_value='gemini335', choices=['gemini335', 'realsense'], description='RGB-D camera driver: Orbbec Gemini 335 or RealSense')
    visualizer_arg = DeclareLaunchArgument('enable_visualizer', default_value='false', description='Show the OpenCV visualization window')

    # -------------------------------------------------------------------------
    # 2. YOLO Human Detector Node Parameters
    # -------------------------------------------------------------------------
    detector_model_arg = DeclareLaunchArgument('detector_model_path', default_value=default_detector_model)
    detector_conf_arg  = DeclareLaunchArgument('detector_confidence_threshold', default_value='0.65', description='YOLO confidence detection threshold')
    detector_nms_arg   = DeclareLaunchArgument('detector_nms_threshold', default_value='0.65', description='YOLO Non-Maximum Suppression threshold')

    # -------------------------------------------------------------------------
    # 3. Human Tracker Node Parameters
    # -------------------------------------------------------------------------
    max_age_arg    = DeclareLaunchArgument('tracker_max_age', default_value='60', description='Max frames to persist a lost track internally')
    max_publish_age_arg = DeclareLaunchArgument('tracker_max_publish_age', default_value='1', description='Max missed tracker updates before hiding an unmatched track')
    min_hits_arg   = DeclareLaunchArgument('tracker_min_hits', default_value='3', description='Min consecutive detections to confirm a track')
    iou_thresh_arg = DeclareLaunchArgument('tracker_iou_threshold', default_value='0.35', description='IoU association threshold for tracking')

    appearance_reid_enabled_arg        = DeclareLaunchArgument('appearance_reid_enabled', default_value='true', description='Enable OSNet appearance-assisted track recovery; false reduces GPU work but may lose IDs after occlusion')
    appearance_match_threshold_arg     = DeclareLaunchArgument('appearance_match_threshold', default_value='0.85', description='Minimum cosine similarity for appearance recovery; higher is stricter and reduces false matches but may miss reappearing people')
    appearance_reid_max_age_arg         = DeclareLaunchArgument('appearance_reid_max_age', default_value='60', description='Maximum missed tracker updates eligible for appearance recovery; 60 is about 2 seconds at 30 Hz; effective lifetime is also limited by tracker_max_age')
    appearance_refresh_interval_arg    = DeclareLaunchArgument('appearance_refresh_interval', default_value='15', description='Eligible successful matches between appearance-gallery refreshes after initial seeding; higher reduces GPU work but adapts more slowly; overlapping crops skip refresh')
    appearance_spatial_base_ratio_arg  = DeclareLaunchArgument('appearance_spatial_base_ratio', default_value='0.10', description='Base recovery search radius around predicted box centre as a fraction of max(image width, height); higher allows larger position errors but more competing people')
    appearance_spatial_speed_ratio_arg = DeclareLaunchArgument('appearance_spatial_speed_ratio_per_sec', default_value='0.75', description='Recovery radius growth per elapsed second since last observation, as a fraction of max(image width, height); radius = base + speed * seconds, capped at 1.0; higher permits faster displacement')

    # -------------------------------------------------------------------------
    # 4. Face & Person Recognizer Node Parameters (SCRFD, AdaFace, OSNet)
    # -------------------------------------------------------------------------
    scrfd_model_arg   = DeclareLaunchArgument('scrfd_model_path', default_value=default_scrfd_model)
    adaface_model_arg = DeclareLaunchArgument('adaface_model_path', default_value=default_adaface_model)
    osnet_model_arg   = DeclareLaunchArgument('osnet_model_path', default_value=default_osnet_model)

    scrfd_conf_arg    = DeclareLaunchArgument('scrfd_conf_threshold', default_value='0.45', description='SCRFD face detection confidence threshold')
    scrfd_nms_arg     = DeclareLaunchArgument('scrfd_nms_threshold', default_value='0.30', description='SCRFD NMS threshold')

    face_thresh_arg   = DeclareLaunchArgument('face_match_threshold', default_value='0.55', description='Face cosine similarity match threshold')
    body_thresh_arg   = DeclareLaunchArgument('body_match_threshold', default_value='0.85', description='Body feature similarity match threshold')

    confirm_frames_arg= DeclareLaunchArgument('min_confirm_frames', default_value='20', description='Min frames required to lock id')
    confirm_dur_arg   = DeclareLaunchArgument('min_confirm_duration_sec', default_value='2.5', description='Min duration required to lock id')

    identity_lock_grace_arg = DeclareLaunchArgument('identity_lock_grace_sec', default_value='3.0', description='Seconds to retain raw-track to persistent-ID lock during temporary occlusion')
    session_reid_timeout_arg = DeclareLaunchArgument('session_reid_timeout_sec', default_value='5.0', description='Seconds to retain anonymous ReID identities after disappearance')
    spatial_gate_age_arg = DeclareLaunchArgument('reid_spatial_gate_max_age_sec', default_value='2.0', description='Maximum age of 3D position used for ReID gating')
    spatial_gate_base_arg = DeclareLaunchArgument('reid_spatial_gate_base_m', default_value='0.75', description='Base allowed 3D displacement during ReID')
    spatial_speed_arg = DeclareLaunchArgument('reid_spatial_gate_max_speed_mps', default_value='2.5', description='Maximum plausible person speed used by ReID spatial gating')
    depth_gate_arg = DeclareLaunchArgument('reid_depth_gate_base_m', default_value='0.75', description='Base allowed depth change during ReID')
    identity_validation_arg = DeclareLaunchArgument( 'identity_validation_interval_sec', default_value='0.5', description='Seconds between deep appearance validation of locked identities')
    # -------------------------------------------------------------------------
    # Topic Definitions
    # -------------------------------------------------------------------------
    image_topic = PythonExpression(["'camera/color/image_raw' if '", LaunchConfiguration('camera_type'), "' == 'gemini335' else 'camera/camera/color/image_raw'"])
    camera_info_topic = PythonExpression(["'camera/color/camera_info' if '", LaunchConfiguration('camera_type'), "' == 'gemini335' else 'camera/camera/color/camera_info'"])
    detections_topic = 'raw_detections'
    tracked_topic    = 'tracked_humans'
    recognized_topic = 'recognized_humans'
    depth_topic = PythonExpression(["'camera/depth/image_raw' if '", LaunchConfiguration('camera_type'), "' == 'gemini335' else 'camera/camera/aligned_depth_to_color/image_raw'"])

    # RealSense Camera Profile Expression ("640x480x30")
    rs_profile = PythonExpression([
        "'", LaunchConfiguration('image_width'), "x",
        LaunchConfiguration('image_height'), "x",
        LaunchConfiguration('framerate'), "'"
    ])

    # -------------------------------------------------------------------------
    # Composable Node Container
    # -------------------------------------------------------------------------
    vision_components = [
            ComposableNode(
                package='opl_human_vision',
                plugin='opl_human_vision::HumanDetectorNode',
                name='human_detector_node',
                parameters=[{
                    'image_topic': image_topic,
                    'detections_topic': detections_topic,
                    'model_path': LaunchConfiguration('detector_model_path'),
                    'confidence_threshold': LaunchConfiguration('detector_confidence_threshold'),
                    'nms_threshold': LaunchConfiguration('detector_nms_threshold')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='opl_human_vision',
                plugin='opl_human_vision::HumanTrackerNode',
                name='human_tracker_node',
                parameters=[{
                    'image_topic': image_topic,
                    'depth_topic': depth_topic,
                    'camera_info_topic': camera_info_topic,
                    'detections_topic': detections_topic,
                    'tracked_topic': tracked_topic,
                    'max_age': LaunchConfiguration('tracker_max_age'),
                    'max_publish_age': LaunchConfiguration('tracker_max_publish_age'),
                    'min_hits': LaunchConfiguration('tracker_min_hits'),
                    'iou_threshold': LaunchConfiguration('tracker_iou_threshold'),
                    'osnet_model_path': LaunchConfiguration('osnet_model_path'),
                    'appearance_reid_enabled': LaunchConfiguration('appearance_reid_enabled'),
                    'appearance_match_threshold': LaunchConfiguration('appearance_match_threshold'),
                    'appearance_reid_max_age': LaunchConfiguration('appearance_reid_max_age'),
                    'appearance_refresh_interval': LaunchConfiguration('appearance_refresh_interval'),
                    'appearance_spatial_base_ratio': LaunchConfiguration('appearance_spatial_base_ratio'),
                    'appearance_spatial_speed_ratio_per_sec': LaunchConfiguration('appearance_spatial_speed_ratio_per_sec')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='opl_human_vision',
                plugin='opl_human_vision::FaceRecognizerNode',
                name='face_recognizer_node',
                parameters=[{
                    'image_topic': image_topic,
                    'tracked_topic': tracked_topic,
                    'recognized_topic': recognized_topic,
                    'scrfd_model_path': LaunchConfiguration('scrfd_model_path'),
                    'adaface_model_path': LaunchConfiguration('adaface_model_path'),
                    'osnet_model_path': LaunchConfiguration('osnet_model_path'),
                    'scrfd_conf_threshold': LaunchConfiguration('scrfd_conf_threshold'),
                    'scrfd_nms_threshold': LaunchConfiguration('scrfd_nms_threshold'),
                    'match_threshold': LaunchConfiguration('face_match_threshold'),
                    'body_match_threshold': LaunchConfiguration('body_match_threshold'),
                    'min_confirm_frames': LaunchConfiguration('min_confirm_frames'),
                    'min_confirm_duration_sec': LaunchConfiguration('min_confirm_duration_sec'),
                    'identity_lock_grace_sec': LaunchConfiguration('identity_lock_grace_sec'),
                    'session_reid_timeout_sec': LaunchConfiguration('session_reid_timeout_sec'),
                    'reid_spatial_gate_max_age_sec': LaunchConfiguration('reid_spatial_gate_max_age_sec'),
                    'reid_spatial_gate_base_m': LaunchConfiguration('reid_spatial_gate_base_m'),
                    'reid_spatial_gate_max_speed_mps': LaunchConfiguration('reid_spatial_gate_max_speed_mps'),
                    'reid_depth_gate_base_m': LaunchConfiguration('reid_depth_gate_base_m'),
                    'identity_validation_interval_sec': LaunchConfiguration('identity_validation_interval_sec')
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        ]

    # Each model owner gets an independent executor. Two workers allow image
    # receipt while its mutually-exclusive inference callback is busy.
    vision_containers = [
        ComposableNodeContainer(
            name=name,
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            parameters=[{'thread_num': 2}],
            composable_node_descriptions=[component],
            output='screen',
        )
        for name, component in zip(
            ['detector_container', 'tracker_container', 'recognizer_container'],
            vision_components,
        )
    ]

    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('camera_type'), "' == 'realsense'"])),
        launch_arguments={
            'align_depth.enable': 'true',
            'depth_module.depth_profile': rs_profile,
            'rgb_camera.color_profile': rs_profile
        }.items()
    )

    gemini_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('opl_gemini335'), 'launch', 'gemini335.launch.py'])),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('camera_type'), "' == 'gemini335'"])),
        launch_arguments={
            'image_width': LaunchConfiguration('image_width'),
            'image_height': LaunchConfiguration('image_height'),
            'framerate': LaunchConfiguration('framerate'),
        }.items(),
    )

    visualizer_node = Node(
        package='opl_human_vision',
        executable='visualizer.py',
        name='pipeline_visualizer',
        parameters=[{
            'image_topic': image_topic
        }],
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        output='screen'
    )

    activator_node = Node(
        package='opl_human_vision',
        executable='lifecycle_activator.py',
        name='lifecycle_activator',
        output='screen'
    )

    return LaunchDescription([
        # Camera Arguments
        width_arg, height_arg, fps_arg, device_arg, camera_type_arg, visualizer_arg,
        # Detector Arguments
        detector_model_arg, detector_conf_arg, detector_nms_arg,
        # Tracker Arguments
        max_age_arg, max_publish_age_arg, min_hits_arg, iou_thresh_arg,
        appearance_reid_enabled_arg, appearance_match_threshold_arg,
        appearance_reid_max_age_arg, appearance_refresh_interval_arg,
        appearance_spatial_base_ratio_arg, appearance_spatial_speed_ratio_arg,
        # Recognizer & Models Arguments
        scrfd_model_arg, adaface_model_arg, osnet_model_arg,
        scrfd_conf_arg, scrfd_nms_arg,
        face_thresh_arg, body_thresh_arg,
        confirm_frames_arg, confirm_dur_arg, identity_lock_grace_arg, session_reid_timeout_arg,
        spatial_gate_age_arg, spatial_gate_base_arg, spatial_speed_arg, depth_gate_arg, identity_validation_arg,
        # Execution Nodes
        *vision_containers, realsense_node, gemini_node, visualizer_node, activator_node
    ])
