import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped # Using TwistStamped for your robot
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class SafetyBrake(Node):
    def __init__(self):
        super().__init__('safety_brake_node')
        
        self.model = YOLO('yolov8n.pt')
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.last_msg = TwistStamped()
        self.frame_count = 0
        
        self.get_logger().info("ADAS System Active: Patrol Mode Engaged.")

    def image_callback(self, msg):
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            self.publisher.publish(self.last_msg)
            return

        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Detect objects
            results = self.model(frame, verbose=False, conf=0.4)
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            threat_detected = False
            
            if len(results[0].boxes) > 0:
                # Check for "Threats"
                for box in results[0].boxes:
                    # Get box size (Height implies distance)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_height = y2 - y1
                    
                    # Logic: If object is TALL (Close), STOP.
                    # Adjust '150' based on how close you want to get
                    if box_height > 150: 
                        threat_detected = True
                        # Visual Warning
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(frame, "BRAKING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        # Object seen but far away - Green Box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # --- CONTROL LOGIC ---
            if threat_detected:
                # EMERGENCY STOP
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 0.0
                self.get_logger().warn("Obstacle Detected! Stopping.")
            else:
                # PATROL: Drive forward slowly
                cmd.twist.linear.x = 0.2
                cmd.twist.angular.z = 0.0

            self.last_msg = cmd
            self.publisher.publish(cmd)
            cv2.imshow("ADAS Camera View", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SafetyBrake()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            stop = TwistStamped()
            node.publisher.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
