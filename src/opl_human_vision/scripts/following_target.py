#!/usr/bin/env python3
"""Service-selected person observations and TF; never command robot motion."""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from opl_interfaces.msg import FollowingTarget, TrackedHumanArray
from opl_interfaces.srv import SelectTarget


class FollowingTargetNode(Node):
    def __init__(self):
        super().__init__('following_target')
        self.selected = 0
        self.identity = 0
        self.name = ''
        self.visible = {}
        self.identities = {}
        self.uncertain = {}
        self.identity_stamps = {}
        self.last_receive = 0.0
        self.last_header = None
        self.reacquire_after = -1
        self.last_tf = None
        self.state = FollowingTarget.UNSELECTED
        self.pub = self.create_publisher(FollowingTarget, 'following_target', 1)
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(TrackedHumanArray, 'tracked_humans', self.tracks, qos_profile_sensor_data)
        self.create_subscription(TrackedHumanArray, 'recognized_humans', self.recognized, qos_profile_sensor_data)
        self.create_service(SelectTarget, 'select_following_target', self.select)
        self.create_timer(0.02, self.watchdog)

    @staticmethod
    def stamp(header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def age(self, header):
        return (self.get_clock().now().nanoseconds - self.stamp(header)) / 1e9

    def fresh(self):
        return (self.last_header is not None and time.monotonic() - self.last_receive <= 0.15
                and 0 <= self.age(self.last_header) <= 0.15)

    def valid_observation(self, human):
        return (human is not None and human.observed and not human.association_ambiguous
                and human.position_valid and self.last_header is not None
                and bool(self.last_header.frame_id) and self.fresh()
                and all(math.isfinite(v) for v in (human.position_3d.x, human.position_3d.y, human.position_3d.z))
                and human.position_3d.z > 0)

    def lose(self, stamp):
        if self.identity and self.reacquire_after < 0:
            self.reacquire_after = stamp

    def select(self, request, response):
        identity = request.identity_id
        if identity == 0:
            self.selected, self.identity, self.name = 0, 0, ''
            self.reacquire_after, self.last_tf = -1, None
            response.success, response.message = True, 'Target cleared; TF disabled'
            self.publish()
            return response
        now = self.get_clock().now().nanoseconds
        matches = [raw for raw, known in self.identities.items()
                   if known[0] == identity and raw in self.visible
                   and 0 <= (now - known[2]) / 1e9 <= 0.75]
        if not 0 < identity < 1000000000 or len(matches) != 1:
            response.success = False
            response.message = 'Enrolled ID must resolve to exactly one fresh visible track'
            return response
        raw = matches[0]
        known = self.identities[raw]
        if (not self.valid_observation(self.visible.get(raw)) or raw in self.uncertain
                or (self.reacquire_after >= 0 and identity == self.identity)):
            response.success = False
            response.message = 'Select a fresh, unambiguous enrolled person with valid depth'
            return response
        self.selected = raw
        self.identity, self.name = known[:2]
        self.reacquire_after = -1
        response.success = True
        response.message = f'Target selected: opl_hr_person_{self.identity}'
        self.publish()
        return response

    def recognized(self, msg):
        if not 0 <= self.age(msg.header) <= 0.75:
            return
        stamp = self.stamp(msg.header)
        candidates = []
        for human in msg.humans:
            raw = human.raw_track_id
            current = self.visible.get(raw)
            if raw <= 0 or current is None or stamp < self.identity_stamps.get(raw, -1):
                continue
            self.identity_stamps[raw] = stamp
            enrolled = (human.observed and not human.association_ambiguous
                        and 0 < human.track_id < 1000000000
                        and human.identity_state in (human.FACE_CONFIRMED, human.TRACKED))
            if human.association_ambiguous or human.identity_state == human.UNCERTAIN:
                self.uncertain[raw] = max(stamp, self.uncertain.get(raw, -1))
            if not enrolled:
                if human.observed:
                    self.identities.pop(raw, None)
                    if raw == self.selected:
                        self.lose(stamp)
                continue
            if raw in self.uncertain:
                if human.identity_state != human.FACE_CONFIRMED or stamp <= self.uncertain[raw]:
                    continue
                del self.uncertain[raw]
            self.identities[raw] = (human.track_id, human.name, stamp)
            if raw == self.selected and human.track_id != self.identity:
                self.lose(stamp)
            if (self.identity and human.track_id == self.identity
                    and human.identity_state == human.FACE_CONFIRMED
                    and stamp > self.reacquire_after and current.observed
                    and not current.association_ambiguous):
                candidates.append(raw)
        if self.reacquire_after >= 0 and len(candidates) == 1 and self.fresh():
            self.selected = candidates[0]
            self.reacquire_after = -1
        self.publish()

    def tracks(self, msg):
        if self.last_header is not None and self.stamp(msg.header) <= self.stamp(self.last_header):
            return
        if self.identity and not self.fresh():
            self.lose(self.get_clock().now().nanoseconds)
        self.last_header = msg.header
        self.last_receive = time.monotonic()
        self.visible = {h.raw_track_id: h for h in msg.humans}
        stamp = self.stamp(msg.header)
        for human in msg.humans:
            if human.association_ambiguous:
                self.uncertain[human.raw_track_id] = max(stamp, self.uncertain.get(human.raw_track_id, -1))
                self.identities.pop(human.raw_track_id, None)
        current = self.visible.get(self.selected)
        if current is None or not current.observed or current.association_ambiguous:
            self.lose(stamp)
        self.uncertain = {k: v for k, v in self.uncertain.items() if k in self.visible or k == self.selected}
        self.identities = {k: v for k, v in self.identities.items() if k in self.visible or k == self.selected}
        self.identity_stamps = {k: v for k, v in self.identity_stamps.items() if k in self.visible or k == self.selected}
        self.publish()

    def publish(self):
        out = FollowingTarget()
        if self.last_header is not None:
            out.header = self.last_header
        out.raw_track_id, out.identity_id, out.name = self.selected, self.identity, self.name
        out.state = FollowingTarget.UNSELECTED if not self.identity else FollowingTarget.LOST
        human = self.visible.get(self.selected)
        known = self.identities.get(self.selected)
        known_fresh = known is not None and 0 <= (self.get_clock().now().nanoseconds - known[2]) / 1e9 <= 0.75
        if self.identity and (not known_fresh or known[0] != self.identity):
            self.lose(self.stamp(out.header))
        if (self.identity and self.reacquire_after < 0 and known_fresh
                and self.selected not in self.uncertain and self.valid_observation(human)):
            out.position_3d = human.position_3d
            out.valid, out.state = True, FollowingTarget.TRACKING
            key = (self.identity, self.stamp(out.header))
            if key != self.last_tf:
                transform = TransformStamped()
                transform.header = out.header
                transform.child_frame_id = f'opl_hr_person_{self.identity}'
                transform.transform.translation.x = human.position_3d.x
                transform.transform.translation.y = human.position_3d.y
                transform.transform.translation.z = human.position_3d.z
                transform.transform.rotation.w = 1.0
                self.broadcaster.sendTransform(transform)
                self.last_tf = key
        self.state = out.state
        self.pub.publish(out)

    def watchdog(self):
        if self.identity and self.state == FollowingTarget.TRACKING and not self.fresh():
            self.lose(self.get_clock().now().nanoseconds)
            self.publish()


def main():
    rclpy.init()
    node = FollowingTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
