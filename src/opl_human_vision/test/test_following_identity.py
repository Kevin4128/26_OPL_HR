import importlib.util
from pathlib import Path
import time
import unittest

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from opl_interfaces.msg import TrackedHuman, TrackedHumanArray, FollowingTarget
from opl_interfaces.srv import SelectTarget

spec = importlib.util.spec_from_file_location('following', Path(__file__).parents[1] / 'scripts/following_target.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FollowingIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = module.FollowingTargetNode()
        self.probe = Node('following_test_probe')
        self.transforms = []
        self.sub = self.probe.create_subscription(TFMessage, '/tf',
            lambda msg: self.transforms.extend(msg.transforms), 100)
        self.a = self.person(8, 2, 'furia')
        self.b = self.person(9, 3, 'other')

    def tearDown(self):
        self.node.destroy_node()
        self.probe.destroy_node()

    def person(self, raw, identity, name):
        h = TrackedHuman(raw_track_id=raw, track_id=identity, name=name,
                         observed=True, position_valid=True, identity_state=TrackedHuman.FACE_CONFIRMED)
        h.position_3d.z = 2.0
        return h

    def message(self, humans):
        msg = TrackedHumanArray(humans=humans)
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        return msg

    def observe(self, humans):
        msg = self.message(humans)
        self.node.tracks(msg)
        self.node.recognized(msg)
        return msg

    def select(self, identity):
        return self.node.select(SelectTarget.Request(identity_id=identity), SelectTarget.Response())

    def collect(self):
        end = time.monotonic() + 0.06
        while time.monotonic() < end:
            rclpy.spin_once(self.probe, timeout_sec=0.005)

    def test_tf_disabled_until_selection_and_cancel(self):
        self.observe([self.a, self.b])
        self.collect()
        self.assertEqual(self.transforms, [])
        self.assertTrue(self.select(2).success)
        self.collect()
        self.assertTrue(self.transforms)
        self.assertEqual({tf.child_frame_id for tf in self.transforms}, {'opl_hr_person_2'})
        self.assertEqual(self.transforms[-1].header.frame_id, 'camera_color_optical_frame')
        self.assertEqual(self.transforms[-1].transform.translation.z, 2.0)
        self.assertEqual(self.transforms[-1].transform.rotation.w, 1.0)
        self.assertTrue(self.select(0).success)
        self.transforms.clear()
        self.observe([self.a, self.b])
        self.collect()
        self.assertEqual(self.transforms, [])
        self.assertEqual(self.node.state, FollowingTarget.UNSELECTED)

    def test_reject_invalid_selection_without_replacing_target(self):
        self.observe([self.a, self.b])
        self.assertTrue(self.select(2).success)
        self.b.position_valid = False
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)
        self.assertFalse(self.select(99).success)
        self.assertFalse(self.select(-1).success)
        self.assertEqual(self.node.identity, 2)
        self.b.position_valid = True
        self.b.track_id = 1000000000
        self.b.identity_state = self.b.UNCONFIRMED
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)

    def test_covered_face_and_depth_recovery(self):
        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.a.identity_state = self.a.TRACKED
        self.a.match_score = 0.0
        msg = self.observe([self.a])
        self.collect()
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.transforms[-1].header.stamp, msg.header.stamp)
        self.a.position_valid = False
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.position_valid = True
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_loss_requires_same_identity_new_face_on_new_raw_track(self):
        old = self.observe([self.a, self.b])
        self.select(2)
        self.observe([self.b])
        self.assertEqual(self.node.identity, 2)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.raw_track_id = 10
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertFalse(self.select(2).success)  # cannot bypass face reacquisition
        self.node.recognized(old)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.identity_state = self.a.FACE_CONFIRMED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.selected, 10)
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.node.identity, 2)

    def test_ambiguity_latches_and_different_face_never_replaces_target(self):
        old = self.observe([self.a, self.b])
        self.select(2)
        self.a.association_ambiguous = True
        self.observe([self.a, self.b])
        self.a.association_ambiguous = False
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.node.recognized(old)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.identity_state = self.a.FACE_CONFIRMED
        self.a.track_id = 3
        self.observe([self.a])
        self.assertEqual(self.node.identity, 2)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.track_id = 2
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_explicit_switch_and_timeout(self):
        self.observe([self.a, self.b])
        self.select(2)
        self.assertTrue(self.select(3).success)
        self.collect()
        self.assertEqual(self.transforms[-1].child_frame_id, 'opl_hr_person_3')
        self.node.last_receive -= 1.0
        self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertEqual(self.node.identity, 3)
        self.b.identity_state = self.b.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)

    def test_no_tf_updates_for_invalid_pose_prediction_or_stale_input(self):
        self.observe([self.a])
        self.select(2)
        self.collect()
        count = len(self.transforms)
        self.assertGreater(count, 0)
        for field, value in [('position_valid', False), ('x', float('nan')),
                             ('z', 0.0), ('observed', False)]:
            self.a.position_valid = True
            self.a.observed = True
            self.a.position_3d.x, self.a.position_3d.z = 0.0, 2.0
            if field in ('x', 'z'):
                setattr(self.a.position_3d, field, value)
            else:
                setattr(self.a, field, value)
            self.observe([self.a])
            self.collect()
            self.assertEqual(len(self.transforms), count)
        self.node.last_receive -= 1.0
        self.node.watchdog()
        self.collect()
        self.assertEqual(len(self.transforms), count)

    def test_select_enrolled_id_resolves_raw_track_and_rejects_raw_id(self):
        self.observe([self.a, self.b])
        self.assertFalse(self.select(8).success)
        self.assertTrue(self.select(2).success)
        self.assertEqual(self.node.selected, 8)
        self.assertEqual(self.node.identity, 2)
        self.assertTrue(self.select(3).success)
        self.assertEqual(self.node.selected, 9)
        self.assertEqual(self.node.identity, 3)

    def test_duplicate_enrolled_id_is_rejected_without_replacing_selection(self):
        self.observe([self.a, self.b])
        self.assertTrue(self.select(2).success)
        self.a.track_id = 3
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)
        self.assertEqual(self.node.identity, 2)
        self.assertFalse(self.select(1000000000).success)
