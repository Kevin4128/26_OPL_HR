#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2


class ObjectVisualizer(Node):
    def __init__(self):
        super().__init__('object_visualizer')
        self.declare_parameter('image_topic', '/head_cam/image')
        self.declare_parameter('detections_topic', '/head_cam/objects')
        image_topic = self.get_parameter('image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        self.bridge = CvBridge()
        self.frame = None
        self.detections = []
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, detections_topic, self.on_detections, qos_profile_sensor_data)
        self.get_logger().info(f'Visualizing {detections_topic} on {image_topic}')

    def on_image(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().error(f'Could not convert camera image: {error}')

    def on_detections(self, msg):
        self.detections = msg.detections

    def draw(self):
        if self.frame is None:
            return
        frame = self.frame.copy()
        for detection in self.detections:
            box = detection.bbox
            x1 = int(box.center.position.x - box.size_x / 2)
            y1 = int(box.center.position.y - box.size_y / 2)
            x2 = int(box.center.position.x + box.size_x / 2)
            y2 = int(box.center.position.y + box.size_y / 2)
            if detection.results:
                hypothesis = detection.results[0].hypothesis
                label = f'{hypothesis.class_id} {hypothesis.score:.2f}'
            else:
                label = 'object'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2)
        cv2.imshow('YOLO11m Object Detection', frame)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectVisualizer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            node.draw()
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
