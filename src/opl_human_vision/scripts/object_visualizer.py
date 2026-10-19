#!/usr/bin/env python3
"""Exact-stamp object annotations; no GUI or inference in this process."""

from collections import OrderedDict
import math
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from opl_interfaces.msg import DetectedObjectArray


def key(header):
    return (header.stamp.sec * 10**9 + header.stamp.nanosec, header.frame_id)


def coordinates(pose):
    values = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
    return '(' + ', '.join(f'{v:.2f}' for v in values) + ')' if all(map(math.isfinite, values)) else None


def annotation_lines(obj):
    lines = [f'{obj.class_name} {obj.confidence:.2f} | {obj.color} {obj.color_confidence:.2f}']
    camera = coordinates(obj.camera_pose) if obj.position_valid else None
    fixed = coordinates(obj.fixed_pose) if obj.position_valid and obj.fixed_position_valid else None
    lines.append(f'camera XYZ [m]: {camera}' if camera else f'Depth rejected: {obj.position_status or "invalid XYZ"}')
    lines.append(f'{obj.fixed_pose.header.frame_id} XYZ [m]: {fixed}' if fixed else
                 f'TF unavailable: {obj.fixed_status or "invalid XYZ"}')
    return lines


def annotate(image, objects):
    output = image.copy()
    height, width = output.shape[:2]
    panels = []
    for obj in objects:
        box = obj.bbox
        x1, y1 = max(0, min(width-1, int(box.x_offset))), max(0, min(height-1, int(box.y_offset)))
        x2, y2 = max(0, min(width-1, int(box.x_offset+box.width))), max(0, min(height-1, int(box.y_offset+box.height)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
        panels.append((x1, y1, annotation_lines(obj)))
    occupied = []
    for x1, y1, lines in panels:
        panel_width = min(width, max(cv2.getTextSize(line[:110], cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0][0] for line in lines)+8)
        left = min(x1, width-panel_width)
        top = max(0, y1-54) if y1 >= 54 else min(y1, max(0, height-54))
        candidates = [(left, top)] + [(x, y) for y in range(0, max(1, height-53), 54)
                                      for x in (0, width-panel_width)]
        for x, y in candidates:
            if not any(x < r and x+panel_width > l and y < b and y+54 > t for l,t,r,b in occupied):
                left, top = x, y
                break
        occupied.append((left, top, left+panel_width, top+54))
        cv2.line(output, (left, top+27), (x1, y1), (0, 180, 200), 1)
        cv2.rectangle(output, (left, top), (left+panel_width, top+54), (0, 0, 0), -1)
        for row, line in enumerate(lines):
            cv2.putText(output, line[:110], (left+4, top+14+row*17), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


class ObjectVisualizer(Node):
    def __init__(self, **kwargs):
        super().__init__('object_visualizer', **kwargs)
        self.declare_parameter('image_topic', '/head_cam/image')
        self.declare_parameter('detections_topic', '/detected_objects')
        self.declare_parameter('annotated_image_topic', '/object_detection/annotated_image')
        self.declare_parameter('visualizer_hz', 7.5)
        self.declare_parameter('buffer_size', 30)
        self.declare_parameter('max_age_s', .5)
        hz = self.get_parameter('visualizer_hz').value
        self.limit = self.get_parameter('buffer_size').value
        self.max_age = self.get_parameter('max_age_s').value
        if not math.isfinite(hz) or not 0 < hz <= 30 or not 1 <= self.limit <= 120 or not math.isfinite(self.max_age) or not 0 < self.max_age <= 5:
            raise ValueError('Invalid visualizer frequency, buffer size or maximum age')
        self.bridge = CvBridge()
        self.images, self.detections = OrderedDict(), OrderedDict()
        self.last_stamp = -1
        self.last_clock = self.get_clock().now().nanoseconds
        self.publisher = self.create_publisher(Image, self.get_parameter('annotated_image_topic').value, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, self.get_parameter('image_topic').value,
                                                 lambda m: self.store(self.images, m), qos_profile_sensor_data)
        self.detection_sub = self.create_subscription(DetectedObjectArray, self.get_parameter('detections_topic').value,
                                                     lambda m: self.store(self.detections, m), qos_profile_sensor_data)
        self.timer = self.create_timer(1.0/hz, self.render, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))

    def check_clock(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_clock:
            self.images.clear()
            self.detections.clear()
            self.last_stamp = -1
        self.last_clock = now
        return now

    def store(self, cache, message):
        self.check_clock()
        stamp, frame = key(message.header)
        if stamp <= self.last_stamp or stamp <= 0 or not frame:
            return
        cache[(stamp, frame)] = (message, time.monotonic())
        cache.move_to_end((stamp, frame))
        while len(cache) > self.limit:
            cache.popitem(last=False)

    def render(self):
        now = self.check_clock()
        wall = time.monotonic()
        for cache in (self.images, self.detections):
            for item, (_, received) in list(cache.items()):
                age = (now-item[0])*1e-9
                if wall-received > self.max_age or age > self.max_age or age < -.02:
                    del cache[item]
        matched = self.images.keys() & self.detections.keys()
        if not matched or self.publisher.get_subscription_count() == 0:
            return
        item = max(matched)
        image, _ = self.images[item]
        detections, _ = self.detections[item]
        try:
            frame = self.bridge.imgmsg_to_cv2(image, 'bgr8')
            message = self.bridge.cv2_to_imgmsg(annotate(frame, detections.objects), 'bgr8')
            message.header = image.header
            self.publisher.publish(message)
            self.last_stamp = item[0]
        except Exception as error:
            self.get_logger().warning(f'Object annotation failed: {error}', throttle_duration_sec=5.0)
        for cache in (self.images, self.detections):
            for old in list(cache):
                if old[0] <= item[0]:
                    del cache[old]


def main():
    rclpy.init()
    node = None
    try:
        node = ObjectVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
