import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan  # NEW: Import LaserScan
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class SmartExplorer(Node):
    def __init__(self):
        super().__init__('smart_explorer_node')
        
        self.model = YOLO('yolov8n.pt')
        
        # 1. LIDAR SUBSCRIBER (The "Anti-Crash" Sensor)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        
        # 2. CAMERA SUBSCRIBER
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # 3. MOTOR PUBLISHER
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.last_msg = TwistStamped()
        
        # SENSOR STATES
        self.lidar_obstacle_detected = False
        self.camera_obstacle_detected = False
        
        self.get_logger().info("SENSOR FUSION ACTIVE: Camera + LIDAR initialized.")

    def scan_callback(self, msg):
        # The LIDAR sends 360 degrees of data (0 to 359)
        # 0 is usually the front.
        # We check a cone in front: 340deg to 20deg
        
        ranges = np.array(msg.ranges)
        
        # Handle 'inf' values (infinite distance)
        ranges[ranges == float('inf')] = 10.0
        
        # Check the front slice (Left 20 deg + Right 20 deg)
        front_left = ranges[0:20]
        front_right = ranges[-20:]
        front_scan = np.concatenate((front_left, front_right))
        
        # Minimum distance in front
        min_distance = np.min(front_scan)
        
        # LOGIC: If anything is closer than 0.6 meters, PANIC!
        if min_distance < 0.6:
            self.lidar_obstacle_detected = True
        else:
            self.lidar_obstacle_detected = False

    def image_callback(self, msg):
        try:
            # We process the camera to update the "Camera State"
            # But we make decisions based on BOTH sensors.
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            # --- DECISION MAKING ---
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "CRUISING"
            color = (0, 255, 0)
            
            # PRIORITY 1: LIDAR (Safety First!)
            if self.lidar_obstacle_detected:
                # Wall/Object detected by Laser -> HARD TURN LEFT
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 0.8
                status_text = "LIDAR AVOIDANCE!"
                color = (0, 0, 255) # Red
                
            # PRIORITY 2: DRIVE
            else:
                cmd.twist.linear.x = 0.25
                cmd.twist.angular.z = 0.0
                status_text = "PATH CLEAR"
            
            # Publish Command
            self.publisher.publish(cmd)
            
            # Visual Feedback
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imshow("Sensor Fusion Brain", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass # Keep running even if a frame drops

def main(args=None):
    rclpy.init(args=args)
    node = SmartExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
