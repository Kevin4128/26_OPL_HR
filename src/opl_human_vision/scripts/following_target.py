#!/usr/bin/env python3
"""Fresh target observations only; never command robot motion."""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
        self.state = FollowingTarget.UNSELECTED
        self.pub = self.create_publisher(FollowingTarget, 'following_target', 1)
        self.create_subscription(TrackedHumanArray, 'tracked_humans', self.tracks, qos_profile_sensor_data)
        self.create_subscription(TrackedHumanArray, 'recognized_humans', self.recognized, qos_profile_sensor_data)
        self.create_service(SelectTarget, 'select_following_target', self.select)
        self.create_timer(0.02, self.watchdog)

    def fresh(self):
        return self.last_header is not None and time.monotonic() - self.last_receive <= 0.15 and 0 <= self.age(self.last_header) <= 0.15

    def age(self, header):
        return (self.get_clock().now().nanoseconds - header.stamp.sec * 10**9 - header.stamp.nanosec) / 1e9

    def select(self, request, response):
        if (not self.fresh() or request.raw_track_id not in self.visible or
                request.raw_track_id in self.uncertain or
                not self.visible[request.raw_track_id].observed or
                self.visible[request.raw_track_id].association_ambiguous):
            response.success = False
            response.message = 'Target must be currently observed in a fresh frame'
            return response
        self.selected = request.raw_track_id
        self.identity, self.name = self.identities.get(self.selected, (0, ''))
        response.success = True
        response.message = 'Target selected'
        self.publish()
        return response

    def recognized(self, msg):
        if not 0 <= self.age(msg.header) <= 0.75:
            return
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        for human in msg.humans:
            raw = human.raw_track_id
            if (raw <= 0 or raw not in self.visible or
                    stamp < self.identity_stamps.get(raw, -1)):
                continue
            self.identity_stamps[raw] = stamp
            if human.identity_state == human.UNCERTAIN or human.association_ambiguous:
                self.uncertain[raw] = max(stamp, self.uncertain.get(raw, -1))
                self.identities.pop(raw, None)
                if raw == self.selected:
                    self.identity, self.name = 0, ''
            elif human.observed and human.identity_state == human.UNCONFIRMED:
                self.identities.pop(raw, None)
                if raw == self.selected:
                    self.identity, self.name = 0, ''
            elif human.observed and human.identity_state in (human.FACE_CONFIRMED, human.TRACKED):
                if raw in self.uncertain:
                    if human.identity_state != human.FACE_CONFIRMED or stamp <= self.uncertain[raw]:
                        continue
                    del self.uncertain[raw]
                self.identities[raw] = (human.track_id, human.name)
                if raw == self.selected:
                    self.identity, self.name = human.track_id, human.name
        if self.state == FollowingTarget.LOST and self.identity and self.fresh():
            candidates = [h.raw_track_id for h in msg.humans if h.identity_state == h.FACE_CONFIRMED and h.observed and not h.association_ambiguous and h.track_id == self.identity and h.raw_track_id in self.visible]
            if len(candidates) == 1:
                self.selected = candidates[0]
        # Publish only on the next fresh tracking observation, never on an identity result.

    def tracks(self, msg):
        self.last_header = msg.header
        self.last_receive = time.monotonic()
        self.visible = {h.raw_track_id: h for h in msg.humans}
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        for human in msg.humans:
            if human.association_ambiguous:
                raw = human.raw_track_id
                self.uncertain[raw] = max(stamp, self.uncertain.get(raw, -1))
                self.identities.pop(raw, None)
                if raw == self.selected:
                    self.identity, self.name = 0, ''
        self.uncertain = {k: v for k, v in self.uncertain.items() if k in self.visible or k == self.selected}
        self.identity_stamps = {k: v for k, v in self.identity_stamps.items() if k in self.visible or k == self.selected}
        self.identities = {k: v for k, v in self.identities.items() if k in self.visible or k == self.selected}
        self.publish()

    def publish(self):
        out = FollowingTarget()
        if self.last_header is not None:
            out.header = self.last_header
        out.raw_track_id, out.identity_id, out.name = self.selected, self.identity, self.name
        out.state = FollowingTarget.UNSELECTED if not self.selected else FollowingTarget.LOST
        human = self.visible.get(self.selected)
        if (self.selected and human is not None and self.fresh() and human.observed and
                human.position_valid and not human.association_ambiguous and
                self.selected not in self.uncertain):
            p = human.position_3d
            if all(math.isfinite(v) for v in (p.x, p.y, p.z)) and p.z > 0:
                out.position_3d = p
                out.valid = True
                out.state = FollowingTarget.TRACKING
        self.state = out.state
        self.pub.publish(out)

    def watchdog(self):
        if self.selected and self.state == FollowingTarget.TRACKING and not self.fresh():
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
