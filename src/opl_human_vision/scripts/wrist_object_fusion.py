#!/usr/bin/env python3
"""Timestamp-synchronized YOLO, color, and wrist depth fusion."""

import time
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from opl_interfaces.msg import DetectedObject, DetectedObjectArray
from rclpy.duration import Duration
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from wrist_perception import (
    classify_color,
    depth_at_box_center,
    depth_to_meters,
    timestamps_within_slop,
    valid_intrinsics,
)


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class WristObjectFusion(Node):
    def __init__(self):
        super().__init__('wrist_object_fusion')
        self.declare_parameter('image_topic', '/wrist_cam/image')
        self.declare_parameter('depth_topic', '/wrist_cam/depth_image')
        self.declare_parameter('camera_info_topic', '/wrist_cam/camera_info')
        self.declare_parameter('detections_topic', '/wrist_cam/objects_raw')
        self.declare_parameter('objects_topic', '/wrist_cam/objects')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('max_sync_slop', 0.03)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('publish_annotated_image', False)
        self.declare_parameter('publish_markers', False)

        self.target_frame = self.get_parameter('target_frame').value
        self.max_sync_slop = float(self.get_parameter('max_sync_slop').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.publish_markers = self.get_parameter('publish_markers').value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.pending = []
        self.tf_poll_timer = self.create_timer(
            0.01, self.process_pending, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.object_pub = self.create_publisher(
            DetectedObjectArray, self.get_parameter('objects_topic').value, 10)
        self.image_pub = self.create_publisher(Image, '~/annotated_image', 2) if self.publish_annotated else None
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10) if self.publish_markers else None

        inputs = [
            Subscriber(self, Image, self.get_parameter('image_topic').value,
                       qos_profile=qos_profile_sensor_data),
            Subscriber(self, Image, self.get_parameter('depth_topic').value,
                       qos_profile=qos_profile_sensor_data),
            Subscriber(self, CameraInfo, self.get_parameter('camera_info_topic').value,
                       qos_profile=qos_profile_sensor_data),
            Subscriber(self, Detection2DArray, self.get_parameter('detections_topic').value,
                       qos_profile=qos_profile_sensor_data),
        ]
        self.sync = ApproximateTimeSynchronizer(
            inputs, queue_size=10, slop=self.max_sync_slop, allow_headerless=False)
        self.sync.registerCallback(self.on_synchronized)
        self.get_logger().info(
            f"Fusing wrist RGB/depth/detections into {self.target_frame}; "
            f"sync tolerance {self.max_sync_slop:.3f}s")

    def publish_empty(self, image, marker_array=None):
        output = DetectedObjectArray()
        output.header.stamp = image.header.stamp
        output.header.frame_id = self.target_frame
        self.object_pub.publish(output)
        if self.marker_pub is not None:
            self.marker_pub.publish(marker_array or MarkerArray())

    def on_synchronized(self, image, depth, camera_info, detections):
        marker_array = MarkerArray()
        if self.marker_pub is not None:
            marker_array.markers.append(Marker(action=Marker.DELETEALL))

        stamps = [stamp_seconds(msg.header) for msg in (image, depth, camera_info, detections)]
        if not timestamps_within_slop(stamps, self.max_sync_slop):
            self.get_logger().warning('Discarding synchronized set with invalid or mismatched timestamps')
            self.publish_empty(image, marker_array)
            return
        frame_ids = [image.header.frame_id, depth.header.frame_id, camera_info.header.frame_id]
        if not frame_ids[0] or any(frame != frame_ids[0] for frame in frame_ids):
            self.get_logger().warning(f'Discarding camera frames that do not match: {frame_ids}')
            self.publish_empty(image, marker_array)
            return
        intrinsics = valid_intrinsics(camera_info)
        if (intrinsics is None or camera_info.width != image.width or
                camera_info.height != image.height or depth.width != image.width or
                depth.height != image.height):
            self.get_logger().warning('Discarding image set with invalid intrinsics or mismatched dimensions')
            self.publish_empty(image, marker_array)
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            depth_m = depth_to_meters(depth)
        except (ValueError, TypeError, cv2.error, CvBridgeError) as error:
            self.get_logger().warning(f'Discarding invalid wrist image data: {error}')
            self.publish_empty(image, marker_array)
            return

        self.pending.append((time.monotonic(), image, detections, rgb, depth_m, intrinsics, marker_array))
        if len(self.pending) > 10:
            _, stale_image, _, _, _, _, stale_markers = self.pending.pop(0)
            self.get_logger().warning('Dropping wrist sample because its TF wait queue is full')
            self.publish_empty(stale_image, stale_markers)
        self.process_pending()

    def process_pending(self):
        if not self.pending:
            return
        queued_at, image, detections, rgb, depth_m, intrinsics, marker_array = self.pending[0]
        stamp = Time.from_msg(image.header.stamp)
        ready = self.tf_buffer.can_transform(
            self.target_frame, image.header.frame_id, stamp, timeout=Duration())
        if not ready:
            if time.monotonic() - queued_at > 0.5:
                self.pending.pop(0)
                self.get_logger().warning('Dropping wrist sample: TF was unavailable at its measurement timestamp')
                self.publish_empty(image, marker_array)
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, image.header.frame_id, stamp)
        except TransformException as error:
            self.pending.pop(0)
            self.get_logger().warning(f'No wrist-camera TF at image timestamp: {error}')
            self.publish_empty(image, marker_array)
            return
        self.pending.pop(0)
        self.publish_transformed(image, detections, rgb, depth_m, intrinsics, transform, marker_array)

    def publish_transformed(self, image, detections, rgb, depth_m, intrinsics, transform, marker_array):
        fx, fy, cx, cy = intrinsics
        output = DetectedObjectArray()
        output.header.stamp = image.header.stamp
        output.header.frame_id = self.target_frame
        annotated = rgb.copy() if self.image_pub is not None else None
        for index, detection in enumerate(detections.detections):
            if not detection.results:
                continue
            hypothesis = max(detection.results, key=lambda result: result.hypothesis.score).hypothesis
            score = float(hypothesis.score)
            box = detection.bbox
            x, y = float(box.center.position.x), float(box.center.position.y)
            width, height = float(box.size_x), float(box.size_y)
            if (not np.isfinite([score, x, y, width, height]).all() or score < 0 or
                    width <= 0 or height <= 0 or not (0 <= x < image.width and 0 <= y < image.height)):
                continue
            z = depth_at_box_center(depth_m, x, y, width, height)
            if z is None or z > self.max_depth_m:
                continue
            u1, u2 = max(0, int(x - width * 0.28)), min(image.width, int(x + width * 0.28))
            v1, v2 = max(0, int(y - height * 0.35)), min(image.height, int(y + height * 0.35))
            color = classify_color(rgb[v1:v2, u1:u2])
            camera_point = PointStamped()
            camera_point.header = image.header
            camera_point.point.x = (x - cx) * z / fx
            camera_point.point.y = (y - cy) * z / fy
            camera_point.point.z = z
            try:
                target_point = do_transform_point(camera_point, transform)
            except (ValueError, TypeError) as error:
                self.get_logger().warning(f'Could not transform object point: {error}')
                continue
            if not np.isfinite([
                    target_point.point.x, target_point.point.y, target_point.point.z]).all():
                continue
            item = DetectedObject()
            item.class_name = hypothesis.class_id
            item.color = color
            item.confidence = score
            item.position = target_point
            output.objects.append(item)

            if annotated is not None:
                cv2.rectangle(annotated, (u1, v1), (u2, v2), (40, 220, 40), 2)
                cv2.putText(annotated, f'{item.color} {item.class_name} {z:.2f}m',
                            (u1, max(18, v1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (40, 220, 40), 1)
            if self.marker_pub is not None:
                marker = Marker()
                marker.header = output.header
                marker.ns = 'wrist_objects'
                marker.id = index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = target_point.point
                marker.pose.orientation.w = 1.0  # Display-only marker; no object orientation is estimated.
                marker.scale.x = marker.scale.y = marker.scale.z = 0.06
                marker.color.a = 0.9
                marker.color.r, marker.color.g, marker.color.b = {
                    'red': (1.0, 0.0, 0.0), 'green': (0.0, 1.0, 0.0),
                    'blue': (0.0, 0.0, 1.0), 'yellow': (1.0, 1.0, 0.0),
                    'unknown': (0.6, 0.6, 0.6),
                }[color]
                marker_array.markers.append(marker)

        self.object_pub.publish(output)
        if self.image_pub is not None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8', header=image.header))
        if self.marker_pub is not None:
            self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = WristObjectFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tf_listener.unregister()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
