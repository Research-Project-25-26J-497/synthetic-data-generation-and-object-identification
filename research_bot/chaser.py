import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# CHANGED: We now import TwistStamped instead of Twist
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ObjectChaser(Node):
    def __init__(self):
        super().__init__('object_chaser_node')
        
        self.get_logger().info("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # CHANGED: Publisher now uses TwistStamped
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile)
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        self.bridge = CvBridge()
        
        # PID Variables
        self.kp = 0.0015
        self.ki = 0.000
        self.kd = 0.01
        self.last_error = 0.0
        self.integral = 0.0
        
        self.frame_count = 0
        
        # CHANGED: Store last command as TwistStamped
        self.last_msg = TwistStamped()
        
        self.get_logger().info("TwistStamped Chaser Node Initialized.")

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            # Republish last command to keep robot alive
            self.publisher.publish(self.last_msg)
            return

        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            height, width, _ = frame.shape
            center_x = width / 2

            results = self.model(frame, verbose=False, conf=0.1)
            
            # CHANGED: Create TwistStamped message
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                object_center_x = (x1 + x2) / 2
                
                # PID Logic
                error_x = center_x - object_center_x
                self.integral += error_x
                derivative = error_x - self.last_error
                
                angular_z = (self.kp * error_x) + (self.ki * self.integral) + (self.kd * derivative)
                
                # Assign to cmd.twist (nested structure)
                cmd.twist.angular.z = max(min(angular_z, 1.0), -1.0)
                cmd.twist.linear.x = 0.3
                
                self.last_error = error_x
                
                # Visuals
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"PID: {cmd.twist.angular.z:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Stop
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 0.5
                self.integral = 0.0
                self.last_error = 0.0

            # Publish
            self.last_msg = cmd
            self.publisher.publish(cmd)
            
            cv2.imshow("Robot Brain (PID Control)", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectChaser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency Stop on Exit
        if rclpy.ok():
            stop_msg = TwistStamped()
            node.publisher.publish(stop_msg)
            node.get_logger().info("Emergency Stop Sent.")
            
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
