"""Real-engine ROS contract test using a saved Gazebo RGB frame and synthetic depth.

This is not live Gazebo accuracy evidence. Run after sourcing the object build:
python3 test_object_node.py --engine PATH --image PATH
"""

import argparse
import copy
import os
import signal
import subprocess
import time

import cv2
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped
from lifecycle_msgs.srv import ChangeState
import numpy as np
from opl_interfaces.msg import DetectedObjectArray
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    assert image is not None
    image = cv2.resize(image, (320, 240))
    rclpy.init()
    node = rclpy.create_node('object_contract_test')
    outputs = []
    tf_metrics = []
    metric_sub = node.create_subscription(DiagnosticArray, '/object_detection/diagnostics',
        lambda m: tf_metrics.append((m.header.frame_id, {v.key: float(v.value) for v in m.status[0].values}))
        if m.status and m.status[0].message == 'ok' else None, 10)
    sub = node.create_subscription(DetectedObjectArray, '/object_test/output', outputs.append, 10)
    publishers = {key: node.create_publisher(typ, '/object_test/' + key, qos_profile_sensor_data)
                  for key, typ in [('rgb', Image), ('depth', Image), ('info', CameraInfo)]}
    command = ['ros2', 'run', 'opl_human_vision', 'object_detection_node_exe', '--ros-args',
               '-r', '__node:=object_contract_detector']
    for key, value in {'model_path': args.engine, 'image_topic': '/object_test/rgb',
                       'depth_topic': '/object_test/depth', 'camera_info_topic': '/object_test/info',
                       'detections_topic': '/object_test/output', 'fixed_frame': 'object_test_fixed',
                       'optical_frame': 'object_test_optical', 'inference_hz': '15.0'}.items():
        command += ['-p', f'{key}:={value}']
    log = open('build/object-contract-node.log', 'w')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    bridge = CvBridge()
    change = node.create_client(ChangeState, '/object_contract_detector/change_state')
    parameters = node.create_client(SetParameters, '/object_contract_detector/set_parameters')

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def transition(number, expected=True):
        assert change.wait_for_service(timeout_sec=15), 'Lifecycle service unavailable'
        request = ChangeState.Request()
        request.transition.id = number
        future = change.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        assert future.done() and future.result().success == expected, f'Transition {number} unexpected result'

    active_optical = 'object_test_optical'
    active_fixed = 'object_test_fixed'
    dynamic = None
    delayed = []
    tf_start_sec = 0

    def trial(mode, seconds=0.65):
        spin(0.25)
        outputs.clear()
        end = time.monotonic() + seconds
        last_publish = 0
        fixed_stamp = node.get_clock().now().to_msg()
        while time.monotonic() < end:
            for item in list(delayed):
                if time.monotonic() >= item[0]:
                    dynamic.sendTransform(item[1])
                    delayed.remove(item)
            if time.monotonic() - last_publish >= 0.1:
                last_publish = time.monotonic()
                stamp = node.get_clock().now().to_msg()
                rgb = bridge.cv2_to_imgmsg(image, 'bgr8')
                rgb.header.stamp = fixed_stamp if mode == 'duplicate' else stamp
                if mode == 'stale':
                    rgb.header.stamp.sec -= 2
                rgb.header.frame_id = active_optical
                if mode == 'optical_frame':
                    rgb.header.frame_id = 'object_test_camera_link'
                depth = bridge.cv2_to_imgmsg(np.full(image.shape[:2], 1.5, np.float32), '32FC1')
                depth.header = copy.deepcopy(rgb.header)
                info = CameraInfo()
                info.header = copy.deepcopy(rgb.header)
                info.height, info.width = image.shape[:2]
                info.k = [475.3, 0., image.shape[1]/2, 0., 475.3, image.shape[0]/2, 0., 0., 1.]
                info.d = [0.] * 5
                if mode == 'depth_frame':
                    depth.header.frame_id = 'unregistered_depth'
                elif mode == 'depth_encoding':
                    depth = bridge.cv2_to_imgmsg(np.ones(image.shape[:2], np.uint8), 'mono8')
                    depth.header = copy.deepcopy(rgb.header)
                elif mode == 'calibration_frame':
                    info.header.frame_id = 'wrong_calibration_frame'
                elif mode == 'calibration':
                    info.k[0] = 0.
                elif mode == 'nonfinite_projection':
                    info.k[0] = 1e-320
                    info.k[2] += 0.25
                elif mode == 'old_depth':
                    depth.header.stamp.sec -= 1
                if mode != 'missing_depth':
                    publishers['depth'].publish(depth)
                if mode != 'missing_info':
                    publishers['info'].publish(info)
                publishers['rgb'].publish(rgb)
                if mode == 'tf_delayed':
                    transform = TransformStamped()
                    transform.header.stamp = copy.deepcopy(rgb.header.stamp)
                    transform.header.frame_id = active_fixed
                    transform.child_frame_id = active_optical
                    transform.transform.rotation.w = 1.
                    transform.transform.translation.x = 10. + rgb.header.stamp.sec-tf_start_sec + rgb.header.stamp.nanosec*1e-9
                    delayed.append((time.monotonic()+.04, transform))
            rclpy.spin_once(node, timeout_sec=0.005)
        return list(outputs)

    try:
        transition(1)
        transition(3)
        spin(0.2)
        normal = trial('normal')
        assert normal and any(m.objects for m in normal), 'Saved image has no actual YOLO detections'
        objects = [o for m in normal for o in m.objects]
        assert any(o.position_valid for o in objects), [(o.class_name, o.position_status) for o in objects]
        assert all(not o.orientation_valid and not o.fixed_position_valid for o in objects)
        assert any(o.fixed_status == 'tf_timeout:tf_missing_frame' for o in objects if o.position_valid)
        assert all(o.position_valid or o.position_status in ('missing_or_unsynchronized_depth',
                   'missing_or_unsynchronized_camera_info') for o in objects)
        expected = {'missing_depth': 'missing_or_unsynchronized_depth',
                    'old_depth': 'missing_or_unsynchronized_depth',
                    'missing_info': 'missing_or_unsynchronized_camera_info',
                    'depth_frame': 'depth_frame_mismatch', 'depth_encoding': 'unsupported_depth_encoding',
                    'calibration': 'invalid_or_nonrectified_calibration',
                    'calibration_frame': 'calibration_frame_mismatch',
                    'nonfinite_projection': 'nonfinite_projection', 'optical_frame': 'unexpected_optical_frame'}
        for mode, reason in expected.items():
            messages = trial(mode)
            objects = [o for m in messages for o in m.objects]
            assert objects, f'{mode}: 2D detections were lost'
            assert any(o.position_status == reason for o in objects), (mode, [o.position_status for o in objects])
            allowed = {reason, 'missing_or_unsynchronized_depth', 'missing_or_unsynchronized_camera_info'}
            assert all(not o.position_valid and o.position_status in allowed and not o.orientation_valid
                       and not o.fixed_position_valid for o in objects), (mode, [o.position_status for o in objects])
            print('PASS', mode, flush=True)
        assert not trial('stale'), 'Stale RGB published'
        assert len(trial('duplicate')) <= 1, 'Duplicate RGB reprocessed'
        broadcaster = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = 'object_test_fixed'
        transform.child_frame_id = 'object_test_optical'
        transform.transform.translation.x = 1.
        transform.transform.rotation.w = 1.
        broadcaster.sendTransform(transform)
        spin(0.2)
        objects = [o for m in trial('normal') for o in m.objects]
        positioned = [o for o in objects if o.position_valid]
        assert positioned and all(o.fixed_position_valid and not o.fixed_orientation_valid and
                                  abs(o.fixed_pose.pose.position.x-o.camera_pose.pose.position.x-1.) < 1e-6
                                  for o in positioned)
        transition(4)
        trial('normal', 0.2)
        assert not outputs, 'Inactive node published detections'
        transition(2)
        transition(1)
        transition(3)
        assert trial('normal'), 'Reactivation failed'
        transition(4)
        transition(2)
        request = SetParameters.Request()
        request.parameters = [Parameter('model_path', value=args.engine + '.missing').to_parameter_msg()]
        future = parameters.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        assert future.done() and all(r.successful for r in future.result().results)
        transition(1, expected=False)
        request.parameters = [Parameter('model_path', value=args.engine).to_parameter_msg()]
        future = parameters.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        assert future.done() and all(r.successful for r in future.result().results)
        transition(1)
        transition(3)
        assert trial('normal'), 'Recovery after failed configure failed'
        for case in ('future', 'past', 'disconnected', 'delayed'):
            transition(4)
            transition(2)
            active_optical = 'object_test_optical_' + case
            active_fixed = 'object_test_fixed_' + case
            request.parameters = [Parameter('optical_frame', value=active_optical).to_parameter_msg(),
                                  Parameter('fixed_frame', value=active_fixed).to_parameter_msg()]
            future = parameters.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            assert future.done() and all(r.successful for r in future.result().results)
            transition(1)
            transition(3)
            spin(.3)
            tf_start_sec = node.get_clock().now().to_msg().sec
            seed = TransformStamped()
            seed.header.stamp = node.get_clock().now().to_msg()
            seed.header.frame_id = active_fixed
            seed.child_frame_id = active_optical
            seed.transform.rotation.w = 1.
            seed.transform.translation.x = 10.
            if case == 'disconnected':
                marker = copy.deepcopy(seed)
                marker.child_frame_id = 'object_test_marker_disconnected'
                seed.header.frame_id = 'object_test_other_root'
                broadcaster.sendTransform([seed, marker])
            else:
                dynamic = TransformBroadcaster(node)
                spin(.2)
                seed.header.stamp = node.get_clock().now().to_msg()
                seed.header.stamp.sec += 5 if case == 'past' else -1
                seed.transform.translation.x = 10. + seed.header.stamp.sec-tf_start_sec + seed.header.stamp.nanosec*1e-9
                second = copy.deepcopy(seed)
                second.header.stamp = rclpy.time.Time(nanoseconds=seed.header.stamp.sec*10**9+seed.header.stamp.nanosec+100000000).to_msg()
                second.transform.translation.x += .1
                dynamic.sendTransform([seed, second])
            spin(.2)
            messages = trial('tf_delayed' if case == 'delayed' else 'normal', 1.2)
            positioned = [(m,o) for m in messages for o in m.objects if o.position_valid]
            assert positioned, case + ': no camera XYZ'
            if case == 'delayed':
                successes = [(m,o) for m,o in positioned if o.fixed_position_valid]
                assert len(successes) >= len(positioned)*.8, 'Late transforms did not recover'
                assert any(values.get('tf_waited_frames', 0) > 0 for frame, values in tf_metrics if frame == active_optical), 'Retry path not exercised'
                for m,o in successes:
                    expected = 10. + m.header.stamp.sec-tf_start_sec + m.header.stamp.nanosec*1e-9
                    assert abs(o.fixed_pose.pose.position.x-o.camera_pose.pose.position.x-expected) < 1e-4, 'Used wrong-time transform'
            else:
                reason = {'future':'tf_future_extrapolation', 'past':'tf_past_extrapolation',
                          'disconnected':'tf_disconnected'}[case]
                assert all(not o.fixed_position_valid for _,o in positioned), case
                assert any(o.fixed_status == 'tf_timeout:'+reason for _,o in positioned), [(o.fixed_status) for _,o in positioned]
            delayed.clear()
            print('PASS TF ' + case, flush=True)
        print('PASS timestamps, TF retry/failure reasons, orientation and lifecycle recovery', flush=True)
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
