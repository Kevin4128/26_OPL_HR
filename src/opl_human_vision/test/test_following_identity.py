import importlib.util
from pathlib import Path
import unittest
import rclpy
from opl_interfaces.msg import TrackedHuman, TrackedHumanArray, FollowingTarget
from opl_interfaces.srv import SelectTarget

spec = importlib.util.spec_from_file_location(
    'following', Path(__file__).parents[1] / 'scripts/following_target.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FollowingIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_retained_identity_and_uncertainty(self):
        node = module.FollowingTargetNode()
        try:
            h = TrackedHuman(raw_track_id=8, track_id=2, name='furia',
                             observed=True, position_valid=True,
                             identity_state=TrackedHuman.TRACKED, match_score=0.0)
            h.position_3d.z = 2.0
            msg = TrackedHumanArray(humans=[h])
            msg.header.stamp = node.get_clock().now().to_msg()
            node.tracks(msg)
            node.recognized(msg)
            response = node.select(SelectTarget.Request(raw_track_id=8), SelectTarget.Response())
            self.assertTrue(response.success)
            self.assertEqual(node.identity, 2)  # score zero means tracked, not unknown
            h.association_ambiguous = True
            h.identity_state = h.UNCERTAIN
            node.recognized(msg)
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.LOST)
            self.assertEqual(node.identity, 0)
            response = node.select(SelectTarget.Request(raw_track_id=8), SelectTarget.Response())
            self.assertFalse(response.success)
            h.association_ambiguous = False
            h.observed = False
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.LOST)
        finally:
            node.destroy_node()

    def test_tracker_ambiguity_latches_until_new_face_confirmation(self):
        node = module.FollowingTargetNode()
        try:
            h = TrackedHuman(raw_track_id=8, track_id=2, name='furia',
                             observed=True, position_valid=True, identity_state=TrackedHuman.FACE_CONFIRMED)
            h.position_3d.z = 2.0
            msg = TrackedHumanArray(humans=[h])
            msg.header.stamp = node.get_clock().now().to_msg()
            node.tracks(msg)
            node.recognized(msg)
            node.select(SelectTarget.Request(raw_track_id=8), SelectTarget.Response())
            old = TrackedHumanArray(humans=[TrackedHuman(raw_track_id=8, track_id=2,
                name='furia', observed=True, identity_state=TrackedHuman.TRACKED)])
            old.header.stamp = msg.header.stamp
            h.association_ambiguous = True
            msg.header.stamp = node.get_clock().now().to_msg()
            node.tracks(msg)
            h.association_ambiguous = False
            msg.header.stamp = node.get_clock().now().to_msg()
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.LOST)
            node.recognized(old)
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.LOST)
            old.humans[0].identity_state = TrackedHuman.FACE_CONFIRMED
            node.recognized(old)  # even a pre-crossing face confirmation is stale
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.LOST)
            msg.header.stamp = node.get_clock().now().to_msg()
            node.recognized(msg)
            node.tracks(msg)
            self.assertEqual(node.state, FollowingTarget.TRACKING)
        finally:
            node.destroy_node()

    def test_observed_unconfirmed_clears_expired_identity(self):
        node = module.FollowingTargetNode()
        try:
            h = TrackedHuman(raw_track_id=8, track_id=2, name='furia',
                             observed=True, position_valid=True, identity_state=TrackedHuman.FACE_CONFIRMED)
            h.position_3d.z = 2.0
            msg = TrackedHumanArray(humans=[h])
            msg.header.stamp = node.get_clock().now().to_msg()
            node.tracks(msg)
            node.recognized(msg)
            node.select(SelectTarget.Request(raw_track_id=8), SelectTarget.Response())
            h.observed = False
            h.identity_state = h.UNCONFIRMED
            msg.header.stamp = node.get_clock().now().to_msg()
            node.recognized(msg)
            self.assertEqual(node.identity, 2)  # a prediction does not expire the lock
            h.observed = True
            msg.header.stamp = node.get_clock().now().to_msg()
            node.recognized(msg)
            self.assertEqual(node.identity, 0)
            self.assertEqual(node.name, '')
        finally:
            node.destroy_node()
