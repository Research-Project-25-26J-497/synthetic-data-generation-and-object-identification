import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class RailMode(Node):
    def __init__(self):
        super().__init__('rail_mode_node')
        
        self.model = YOLO('yolov8n.pt')
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.last_msg = TwistStamped()
        
        # SLOW AND STEADY
        self.drive_speed = 0.15 
        
        self.get_logger().info("RAIL MODE ACTIVE: Steering Locked. Moving Straight.")

    def image_callback(self, msg):
        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Detect objects (Low confidence to catch everything)
            results = self.model(frame, verbose=False, conf=0.15)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            stop_triggers = False
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    obj_height = y2 - y1
                    
                    # DEBUG: Print size to help you position the cone
                    # Look at your terminal! If it says "80px" and doesn't stop, 
                    # change the 70 below to 60.
                    print(f"Object Size: {obj_height:.1f} px")
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    # STOP THRESHOLD
                    # If the object is bigger than 70 pixels, STOP.
                    if obj_height > 120: 
                        stop_triggers = True
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(frame, "EMERGENCY STOP", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            # --- THE RAIL LOGIC ---
            if stop_triggers:
                cmd.twist.linear.x = 0.0  # BRAKE
            else:
                cmd.twist.linear.x = self.drive_speed # GO STRAIGHT
            
            # LOCK STEERING (Prevents Wobble)
            cmd.twist.angular.z = 0.0

            self.publisher.publish(cmd)
            cv2.imshow("Rail Mode View", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = RailMode()
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
