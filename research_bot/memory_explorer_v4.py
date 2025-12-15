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

class FinalMemoryExplorer(Node):
    def __init__(self):
        super().__init__('final_memory_explorer')
        
        self.model = YOLO('yolov8n.pt')
        
        # SENSORS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # STATE VARIABLES
        self.lidar_ranges = []
        self.obstacle_detected = False
        self.history = [] 
        self.current_x = 0.0
        self.current_y = 0.0
        self.frame_counter = 0
        self.front_dist = 10.0 # Track distance to front object
        
        # INTELLIGENT ESCAPE STATE
        self.escape_timer = 0
        self.target_angle = 0.0 
        self.memory_cooldown = 0 # "Amnesia" timer
        
        self.get_logger().info("FINAL EXPLORER: Safety Overrides Enabled.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Front collision check (Narrow Cone)
        front_indices = np.concatenate((np.arange(0, 15), np.arange(345, 360)))
        self.front_dist = np.min(ranges[front_indices])
        
        # If anything is closer than 0.6m, flag it
        if self.front_dist < 0.6:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def find_largest_gap(self):
        """ Scans for the most open quadrant """
        if len(self.lidar_ranges) == 0: return 1.0
        
        # Split into 4 Quadrants
        q1 = np.mean(self.lidar_ranges[0:90])    # Front Left
        q2 = np.mean(self.lidar_ranges[90:180])  # Back Left
        q3 = np.mean(self.lidar_ranges[180:270]) # Back Right
        q4 = np.mean(self.lidar_ranges[270:360]) # Front Right
        
        gaps = [q1, q2, q3, q4]
        best_quadrant = gaps.index(max(gaps))
        
        # Turn towards the best open space
        if best_quadrant == 0 or best_quadrant == 1:
            return 0.9  # Turn Left
        else:
            return -0.9 # Turn Right

    def check_memory(self):
        # Don't check memory if we are in "Cooldown" (Just escaped)
        if self.memory_cooldown > 0:
            self.memory_cooldown -= 1
            return False

        if len(self.history) < 50: return False
        
        # Only check history older than 10 seconds (prevents self-fear)
        # 30 frames per sec approx -> 300 frames
        limit = min(len(self.history), 300)
        old_history = self.history[:-limit]
        
        for (past_x, past_y) in old_history:
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            if dist < 0.8: return True # Increased detection radius slightly
        return False

    def image_callback(self, msg):
        try:
            # 1. Update Memory (Less frequent = Less Panic)
            self.frame_counter += 1
            if self.frame_counter % 30 == 0: # Only drop crumb every 30 frames
                self.history.append((self.current_x, self.current_y))
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "EXPLORING"
            color = (0, 255, 0)
            
            # --- DECISION TREE ---
            
            # STATE 1: ESCAPE MODE (But with Safety)
            if self.escape_timer > 0:
                # SAFETY OVERRIDE: If we are about to hit the cylinder, ABORT!
                if self.front_dist < 0.5:
                    self.escape_timer = 0 # Cancel Escape
                    self.memory_cooldown = 100 # Don't check memory for a bit
                    # Let normal avoidance take over immediately
                    cmd.twist.linear.x = 0.0
                    cmd.twist.angular.z = 1.0 # Emergency spin
                    status_text = "ESCAPE ABORTED (WALL)"
                    color = (0, 0, 255)
                else:
                    # Proceed with Calculated Escape
                    self.escape_timer -= 1
                    cmd.twist.linear.x = 0.15
                    cmd.twist.angular.z = self.target_angle
                    status_text = f"SMART ESCAPE ({self.escape_timer})"
                    color = (255, 0, 255) # Purple
                
            # STATE 2: OBSTACLE AVOIDANCE (Highest Standard Priority)
            elif self.obstacle_detected:
                cmd.twist.linear.x = 0.0
                # Use Gap Finder for avoidance too
                cmd.twist.angular.z = self.find_largest_gap()
                status_text = "OBSTACLE AVOIDANCE"
                color = (0, 0, 255)

            # STATE 3: DEJA VU CHECK
            elif self.check_memory():
                # Loop detected!
                self.target_angle = self.find_largest_gap()
                self.escape_timer = 120 # Commit to escape for 4 seconds
                self.memory_cooldown = 200 # Grant immunity for after the escape
                
                self.get_logger().warn(f"Loop Detected! Escaping towards gap.")

            # STATE 4: CRUISE
            else:
                cmd.twist.linear.x = 0.22
                cmd.twist.angular.z = 0.0
                status_text = "PATH CLEAR"

            self.publisher.publish(cmd)
            
            # Visuals
            cv2.putText(frame, f"Mem: {len(self.history)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            if self.memory_cooldown > 0:
                cv2.putText(frame, "AMNESIA ACTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Final Memory Explorer", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = FinalMemoryExplorer()
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
