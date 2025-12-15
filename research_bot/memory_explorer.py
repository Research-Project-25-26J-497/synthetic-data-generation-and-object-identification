import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry  # <--- NEW: To track position
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import math

class MemoryExplorer(Node):
    def __init__(self):
        super().__init__('memory_explorer_node')
        
        self.model = YOLO('yolov8n.pt')
        
        # SENSORS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # NEW: Listen to Odometer to know X,Y position
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # STATE & MEMORY
        self.lidar_ranges = []
        self.obstacle_detected = False
        self.history = []  # List of (x, y) tuples
        self.current_x = 0.0
        self.current_y = 0.0
        self.frame_counter = 0
        self.retrace_cooldown = 0
        
        self.get_logger().info("MEMORY EXPLORER: I will remember where I have been.")

    def odom_callback(self, msg):
        # Update current position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Safety Check (Front collision)
        front_indices = np.concatenate((np.arange(0, 15), np.arange(345, 360)))
        if np.min(ranges[front_indices]) < 0.65:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def check_memory(self):
        """
        Returns TRUE if we are currently standing in a spot we visited a while ago.
        """
        if len(self.history) < 100:
            return False # Not enough memory yet
            
        # Ignore the last 50 points (recent history) so we don't fear ourselves
        old_history = self.history[:-100]
        
        for (past_x, past_y) in old_history:
            # Euclidean Distance Formula
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            
            # If we are within 0.5 meters of an old spot
            if dist < 0.5:
                return True
        return False

    def find_safest_direction(self):
        if len(self.lidar_ranges) == 0: return 0.0
        
        # Determine "Gap" direction (Left vs Right Openness)
        left_avg = np.mean(self.lidar_ranges[0:90])
        right_avg = np.mean(self.lidar_ranges[270:360])
        
        if left_avg > right_avg:
            return 0.6 
        else:
            return -0.6

    def image_callback(self, msg):
        try:
            # 1. Update Memory (Every 10 frames approx)
            self.frame_counter += 1
            if self.frame_counter % 10 == 0:
                self.history.append((self.current_x, self.current_y))
            
            # Process Visuals
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "EXPLORING"
            color = (0, 255, 0)
            
            # --- INTELLIGENT NAVIGATION STACK ---
            
            # PRIORITY 1: RETRACE AVOIDANCE (The "Novelty" Check)
            if self.retrace_cooldown > 0:
                # We are currently executing a "Get Away" maneuver
                self.retrace_cooldown -= 1
                cmd.twist.linear.x = 0.1
                cmd.twist.angular.z = 0.9 # Hard spin
                status_text = "REROUTING..."
                color = (255, 0, 255) # Purple
                
            elif self.check_memory():
                # We realized we are in an old spot! Trigger the maneuver.
                self.retrace_cooldown = 40 # Spend 40 frames turning away
                self.get_logger().warn("DEJA VU! I have been here before. Rerouting.")
            
            # PRIORITY 2: COLLISION AVOIDANCE
            elif self.obstacle_detected:
                cmd.twist.linear.x = 0.0
                turn = self.find_safest_direction()
                cmd.twist.angular.z = turn
                status_text = "AVOIDING"
                color = (0, 0, 255)
                
            # PRIORITY 3: CRUISE
            else:
                cmd.twist.linear.x = 0.25
                cmd.twist.angular.z = 0.0
                status_text = "NEW PATH"

            self.publisher.publish(cmd)
            
            # Draw Path on Screen (Visualization)
            # This creates a "Mini Map" on the camera feed showing crumbs
            if len(self.history) > 2:
                # Scale coordinates for visualization (Rough approximation)
                for i in range(1, len(self.history), 5): 
                    pt = self.history[i]
                    # We can't easily draw world coords on screen without math, 
                    # so we just show the Status Text as the indicator.
                    pass
            
            cv2.putText(frame, f"Memory Nodes: {len(self.history)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Memory Explorer", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MemoryExplorer()
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
