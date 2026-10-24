#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from rclpy.qos import qos_profile_sensor_data

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        self.declare_parameter('image_topic', 'camera/image_raw')
        self.declare_parameter('video_device', 0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('framerate', 30.0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        device_param = self.get_parameter('video_device').value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        fps = self.get_parameter('framerate').get_parameter_value().double_value

        try:
            video_device = int(device_param)
        except ValueError:
            video_device = str(device_param)

        # FIX: Publish using SensorDataQoS so the AI nodes don't backlog old frames
        self.publisher_ = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self.bridge = CvBridge()

        self.video_device = video_device
        self.width, self.height, self.fps = width, height, fps
        self.reconnect_attempts = 0
        self.failed_reads = 0
        self.exhausted = False
        self.next_reconnect = None
        self.cap = None
        self.open_camera()

        timer_period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def open_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.video_device)
        except cv2.error as error:
            self.cap = None
            self.get_logger().error(f'Camera open failed: {error}')
            self.schedule_reconnect()
            return
        if self.cap.isOpened():
            self.next_reconnect = None
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(f"Webcam initialized on [{self.video_device}] @ {actual_w}x{actual_h}")
        else:
            self.get_logger().error(f"Could not open webcam device [{self.video_device}]!")

            self.schedule_reconnect()

    def schedule_reconnect(self):
        if self.cap is not None:
            self.cap.release()
        self.failed_reads = 0
        if self.reconnect_attempts >= 5:
            self.exhausted = True
            self.get_logger().error('Camera recovery exhausted after five attempts; restart the node')
        else:
            self.next_reconnect = time.monotonic() + 1.0

    def timer_callback(self):
        if self.exhausted:
            return
        if self.cap is None or not self.cap.isOpened():
            if self.next_reconnect is None:
                self.schedule_reconnect()
            elif time.monotonic() >= self.next_reconnect:
                self.reconnect_attempts += 1
                self.open_camera()
            return

        try:
            ret, frame = self.cap.read()
        except cv2.error as error:
            self.get_logger().warning(f'Camera read failed: {error}')
            ret, frame = False, None
        if ret and frame is not None and frame.size:
            self.failed_reads = 0
            self.reconnect_attempts = 0
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.publisher_.publish(msg)
        else:
            self.failed_reads += 1
            if self.failed_reads >= 3 or not self.cap.isOpened():
                self.get_logger().warning('Camera reads failed or the device closed; reconnecting')
                self.schedule_reconnect()

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
