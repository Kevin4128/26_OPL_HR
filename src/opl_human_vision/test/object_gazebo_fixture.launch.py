"""Optional live verification fixture using the existing robot/world, without navigation."""

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    robot = get_package_share_directory('wheeltec_robot_urdf')
    bringup = get_package_share_directory('rtab_bringup')
    document = xacro.process_file(os.path.join(robot, 'urdf', 'opl_simple.urdf'))
    gazebo = document.createElement('gazebo')
    static = document.createElement('static')
    static.appendChild(document.createTextNode('true'))
    gazebo.appendChild(static)
    document.documentElement.appendChild(gazebo)
    description = document.toxml()
    topics = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/head_cam/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/head_cam/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/head_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    ]
    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(robot) + ':' + os.path.join(bringup, 'gazebo', 'models')),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', os.path.join(bringup, 'gazebo', 'test.sdf')], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description, 'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=topics,
             parameters=[{'use_sim_time': True}]),
        Node(package='robot_localization', executable='ekf_node', name='object_fixture_ekf',
             parameters=[{'use_sim_time': True, 'two_d_mode': True, 'frequency': 30.0,
                          'odom_frame': 'odom', 'base_link_frame': 'base_footprint', 'world_frame': 'odom',
                          'odom0': '/odom', 'odom0_config': [False, False, False, False, False, False,
                                                         True, True, False, False, False, True, False, False, False]}]),
        Node(package='ros_gz_sim', executable='create',
             arguments=['-topic', 'robot_description', '-name', 'wheeltec_oplbot',
                        '-x', '2.15', '-y', '-0.1', '-z', '0.0', '-Y', '-1.0'], output='screen'),
    ])
