import copy
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rclpy
from opl_interfaces.msg import DetectedObject, DetectedObjectArray

spec = importlib.util.spec_from_file_location('object_visualizer', Path(__file__).parents[1]/'scripts/object_visualizer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ObjectVisualizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = module.ObjectVisualizer()
        self.node.publisher = Mock()
        self.node.publisher.get_subscription_count.return_value = 1
        self.image = self.node.bridge.cv2_to_imgmsg(np.zeros((100, 120, 3), np.uint8), 'bgr8')
        self.image.header.stamp = self.node.get_clock().now().to_msg()
        self.image.header.frame_id = 'test_optical'
        self.array = DetectedObjectArray()
        self.array.header = copy.deepcopy(self.image.header)

    def tearDown(self):
        self.node.destroy_node()

    def test_exact_timestamp_and_frame_only(self):
        self.node.store(self.node.images, self.image)
        self.array.header.stamp.nanosec += 1
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_not_called()
        self.array.header = copy.deepcopy(self.image.header)
        self.array.header.frame_id = 'wrong_frame'
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_not_called()
        self.array.header = copy.deepcopy(self.image.header)
        self.node.store(self.node.detections, self.array)
        self.node.render()
        message = self.node.publisher.publish.call_args.args[0]
        self.assertEqual(message.header, self.image.header)
        self.assertEqual(message.data, self.image.data)  # Empty array clears old boxes.

    def test_invalid_positions_never_show_placeholders(self):
        obj = DetectedObject()
        obj.class_name, obj.color, obj.confidence, obj.color_confidence = 'bottle', 'red', .9, .8
        obj.position_status, obj.fixed_status = 'ambiguous_depth', 'tf_timeout:tf_future_extrapolation'
        lines = module.annotation_lines(obj)
        self.assertIn('bottle 0.90 | red 0.80', lines[0])
        self.assertIn('Depth rejected: ambiguous_depth', lines[1])
        self.assertIn('TF unavailable', lines[2])
        self.assertNotIn('(0.00', ' '.join(lines))
        obj.position_valid = True
        obj.camera_pose.pose.position.z = 1.5
        self.assertIn('1.50', module.annotation_lines(obj)[1])
        self.assertIn('TF unavailable', module.annotation_lines(obj)[2])
        obj.fixed_position_valid = True
        obj.fixed_pose.header.frame_id = 'odom'
        obj.fixed_pose.pose.position.x = 2.
        self.assertIn('odom XYZ [m]: (2.00', module.annotation_lines(obj)[2])
        obj.camera_pose.pose.position.x = float('nan')
        self.assertIn('Depth rejected', module.annotation_lines(obj)[1])

    def test_buffers_are_bounded_and_old_images_rejected(self):
        for i in range(80):
            image = copy.deepcopy(self.image)
            image.header.stamp.nanosec += i
            self.node.store(self.node.images, image)
        self.assertEqual(len(self.node.images), self.node.limit)
        self.node.images.clear()
        self.image.header.stamp.sec -= 1
        self.array.header = copy.deepcopy(self.image.header)
        self.node.store(self.node.images, self.image)
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_not_called()

    def test_late_image_matches_and_is_published_only_once(self):
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_not_called()
        self.node.store(self.node.images, self.image)
        self.node.render()
        self.node.store(self.node.images, self.image)
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_called_once()
        older = copy.deepcopy(self.image)
        older.header.stamp = rclpy.time.Time(nanoseconds=module.key(older.header)[0]-1000000).to_msg()
        older_array = copy.deepcopy(self.array)
        older_array.header = copy.deepcopy(older.header)
        self.node.store(self.node.images, older)
        self.node.store(self.node.detections, older_array)
        self.node.render()
        self.node.publisher.publish.assert_called_once()

    def test_clock_reset_clears_old_epoch(self):
        self.node.store(self.node.images, self.image)
        self.node.store(self.node.detections, self.array)
        self.node.last_stamp = module.key(self.image.header)[0]
        self.node.last_clock = self.node.get_clock().now().nanoseconds+10**10
        self.node.check_clock()
        self.assertFalse(self.node.images)
        self.assertFalse(self.node.detections)
        self.assertEqual(self.node.last_stamp, -1)

    def test_annotation_failure_does_not_stop_publisher(self):
        self.node.store(self.node.images, self.image)
        self.node.store(self.node.detections, self.array)
        with patch.object(module, 'annotate', side_effect=ValueError('test failure')):
            self.node.render()
        self.node.store(self.node.images, self.image)
        self.node.store(self.node.detections, self.array)
        self.node.render()
        self.node.publisher.publish.assert_called_once()

    def test_boxes_and_text_are_drawn(self):
        obj = DetectedObject()
        obj.bbox.x_offset, obj.bbox.y_offset, obj.bbox.width, obj.bbox.height = 10, 60, 30, 30
        obj.class_name = 'cup'
        rendered = module.annotate(np.zeros((100, 120, 3), np.uint8), [obj])
        self.assertGreater(np.count_nonzero(rendered), 50)
