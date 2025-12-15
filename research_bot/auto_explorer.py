import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class AutoExplorer(Node):
    def __init__(self):
        super().__init__('auto_explorer_node')
        
        self.model = YOLO('yolov8n.pt')
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.last_msg = TwistStamped()
        
        # DEFAULT CRUISE SPEED
        self.cruise_speed = 0.20 
        
        self.get_logger().info("AUTO EXPLORER ACTIVE: I will drive and avoid obstacles.")

    def image_callback(self, msg):
        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Detect objects (Low confidence to catch walls/pillars)
            results = self.model(frame, verbose=False, conf=0.15)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            obstacle_close = False
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    obj_height = y2 - y1
                    
                    # VISUALIZE
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    # LOGIC: If object is bigger than 90px, it's too close -> AVOID
                    if obj_height > 90: 
                        obstacle_close = True
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(frame, "AVOIDING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            # --- EXPLORER LOGIC ---
            if obstacle_close:
                # EVASIVE MANEUVER: Slow down and Turn Left
                cmd.twist.linear.x = 0.05
                cmd.twist.angular.z = 0.6
            else:
                # CLEAR PATH: Drive Straight
                cmd.twist.linear.x = self.cruise_speed
                cmd.twist.angular.z = 0.0

            self.publisher.publish(cmd)
            cv2.imshow("Explorer View", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = AutoExplorer()
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
