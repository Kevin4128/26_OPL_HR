import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import rclpy

spec = importlib.util.spec_from_file_location('webcam', Path(__file__).parents[1] / 'scripts/webcam_publisher.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Capture:
    def __init__(self, opened, frames=()):
        self.opened = opened
        self.frames = iter(frames)
        self.settings = {}
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings[key] = value

    def get(self, key):
        return self.settings.get(key, 0)

    def read(self):
        return next(self.frames, (False, None))

    def release(self):
        self.released = True
        self.opened = False


class WebcamReconnectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_open_failure_retries_once_a_second_and_stops_after_five(self):
        now = [0.0]
        captures = []
        def create(device):
            capture = Capture(False)
            captures.append(capture)
            return capture
        with patch.object(module.cv2, 'VideoCapture', side_effect=create), patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            node = module.WebcamPublisher()
            try:
                for t in (0.0, 0.2, 0.99, 1.0, 1.01, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0):
                    now[0] = t
                    node.timer_callback()
                self.assertEqual(len(captures), 6)  # initial open + five reconnects
                self.assertTrue(all(c.released for c in captures))
            finally:
                node.destroy_node()

    def test_read_failure_recovery_reapplies_settings_and_resets_only_on_frame(self):
        now = [0.0]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        first = Capture(True, [(False, None)] * 3)
        second = Capture(True, [(False, None)] * 3)
        third = Capture(True, [(True, frame)] + [(False, None)] * 3)
        captures = [first, second, third]
        with patch.object(module.cv2, 'VideoCapture', side_effect=captures) as factory, patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            node = module.WebcamPublisher()
            try:
                node.timer_callback(); node.timer_callback()
                self.assertFalse(first.released)
                node.timer_callback()
                self.assertTrue(first.released)
                node.timer_callback()
                self.assertEqual(factory.call_count, 1)
                now[0] = 1.0
                node.timer_callback()
                self.assertEqual(node.reconnect_attempts, 1)
                for _ in range(3):
                    node.timer_callback()
                now[0] = 2.0
                node.timer_callback()
                self.assertEqual(node.reconnect_attempts, 2)  # opening is not recovery
                node.timer_callback()
                self.assertEqual(node.reconnect_attempts, 0)
                for capture in captures:
                    self.assertEqual(capture.settings[module.cv2.CAP_PROP_FRAME_WIDTH], 640)
                    self.assertEqual(capture.settings[module.cv2.CAP_PROP_FRAME_HEIGHT], 480)
                    self.assertEqual(capture.settings[module.cv2.CAP_PROP_FPS], 30.0)
            finally:
                node.destroy_node()

    def test_success_between_failures_breaks_read_failure_streak(self):
        now = [0.0]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        capture = Capture(True, [(False, None), (False, None), (True, frame),
                                 (False, None), (False, None), (False, None)])
        with patch.object(module.cv2, 'VideoCapture', return_value=capture), patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            node = module.WebcamPublisher()
            try:
                for _ in range(5):
                    node.timer_callback()
                self.assertFalse(capture.released)
                node.timer_callback()
                self.assertTrue(capture.released)
            finally:
                node.destroy_node()

    def test_driver_errors_use_the_same_bounded_retry_path(self):
        now = [0.0]
        with patch.object(module.cv2, 'VideoCapture', side_effect=module.cv2.error('disconnected')) as factory, patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            node = module.WebcamPublisher()
            try:
                for t in range(1, 10):
                    now[0] = float(t)
                    node.timer_callback()
                self.assertEqual(factory.call_count, 6)
                self.assertTrue(node.exhausted)
            finally:
                node.destroy_node()

    def test_backend_closing_handle_cannot_bypass_reconnect_delay(self):
        class ClosingCapture(Capture):
            def read(self):
                self.opened = False
                return False, None
        now = [0.0]
        with patch.object(module.cv2, 'VideoCapture', side_effect=lambda _: ClosingCapture(True)) as factory, patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            node = module.WebcamPublisher()
            try:
                for _ in range(20):
                    node.timer_callback()
                self.assertEqual(factory.call_count, 1)
                now[0] = 1.0
                for _ in range(20):
                    node.timer_callback()
                self.assertEqual(factory.call_count, 2)
                now[0] = 2.0
                node.timer_callback()
                self.assertEqual(factory.call_count, 3)
            finally:
                node.destroy_node()
