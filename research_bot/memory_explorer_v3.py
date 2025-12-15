import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import math

class SmartMemoryExplorer(Node):
    def __init__(self):
        super().__init__('smart_memory_explorer')
        
        self.model = YOLO('yolov8n.pt')
        
        # SENSORS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # STATE
        self.lidar_ranges = []
        self.history = [] 
        self.current_x = 0.0
        self.current_y = 0.0
        self.frame_counter = 0
        
        # INTELLIGENT ESCAPE STATE
        self.escape_active = False
        self.escape_timer = 0
        self.target_angle = 0.0 # The calculated "Best Path" to follow
        
        self.get_logger().info("SMART MEMORY EXPLORER: Calculating optimal escape vectors.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

    def find_largest_gap(self):
        """
        Scans all 360 degrees to find the 'Deepest' open space.
        Returns: +1.0 (Turn Left) or -1.0 (Turn Right) based on where the gap is.
        """
        if len(self.lidar_ranges) == 0: return 1.0
        
        # Split LIDAR into 4 Quadrants
        # Front-Left (0-90), Back-Left (90-180), Back-Right (180-270), Front-Right (270-360)
        q1 = np.mean(self.lidar_ranges[0:90])    # Front Left
        q2 = np.mean(self.lidar_ranges[90:180])  # Back Left
        q3 = np.mean(self.lidar_ranges[180:270]) # Back Right
        q4 = np.mean(self.lidar_ranges[270:360]) # Front Right
        
        # Find the max open space
        gaps = [q1, q2, q3, q4]
        best_quadrant = gaps.index(max(gaps))
        
        # Logic: If the best space is on the Left (Q1/Q2), turn Left.
        # If best space is Right (Q3/Q4), turn Right.
        if best_quadrant == 0 or best_quadrant == 1:
            return 0.8  # Turn Left towards open space
        else:
            return -0.8 # Turn Right towards open space

    def check_memory(self):
        if len(self.history) < 50: return False
        old_history = self.history[:-50]
        for (past_x, past_y) in old_history:
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            if dist < 0.6: return True
        return False

    def image_callback(self, msg):
        try:
            self.frame_counter += 1
            if self.frame_counter % 5 == 0:
                self.history.append((self.current_x, self.current_y))
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "EXPLORING"
            color = (0, 255, 0)
            
            # --- INTELLIGENT BEHAVIOR ---
            
            # 1. ESCAPE MODE (High Priority)
            if self.escape_timer > 0:
                self.escape_timer -= 1
                
                # Use the PRE-CALCULATED optimal direction
                cmd.twist.linear.x = 0.15  # Controlled speed (Safe)
                cmd.twist.angular.z = self.target_angle
                
                status_text = f"SMART ESCAPE ({self.escape_timer})"
                color = (255, 0, 255) # Purple
                
            # 2. TRIGGER ESCAPE (Deja Vu Detected)
            elif self.check_memory():
                # STOP and THINK
                self.target_angle = self.find_largest_gap() # <--- The "Calculated" decision
                self.escape_timer = 100 # Lock this path for 3 seconds
                
                self.get_logger().warn(f"Loop Detected! Calculated Escape Vector: {self.target_angle}")
                
            # 3. CRUISE MODE
            else:
                # Basic obstacle avoidance for cruising
                if len(self.lidar_ranges) > 0:
                     # Simple safety check
                    front = np.concatenate((self.lidar_ranges[0:20], self.lidar_ranges[340:360]))
                    if np.min(front) < 0.75:
                        # Standard Avoidance
                        cmd.twist.linear.x = 0.0
                        # Turn towards the larger open side (Left or Right)
                        left = np.mean(self.lidar_ranges[0:90])
                        right = np.mean(self.lidar_ranges[270:360])
                        cmd.twist.angular.z = 0.6 if left > right else -0.6
                        status_text = "AVOIDING"
                        color = (0, 0, 255)
                    else:
                        # All clear
                        cmd.twist.linear.x = 0.2
                        cmd.twist.angular.z = 0.0
                        status_text = "PATH CLEAR"

            self.publisher.publish(cmd)
            
            cv2.putText(frame, f"History: {len(self.history)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            cv2.imshow("Smart Memory V3", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SmartMemoryExplorer()
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
