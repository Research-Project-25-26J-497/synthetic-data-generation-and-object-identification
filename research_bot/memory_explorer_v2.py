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
import random

class MemoryExplorerV2(Node):
    def __init__(self):
        super().__init__('memory_explorer_v2')
        
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
        
        # ESCAPE MECHANISM
        self.escape_timer = 0
        self.escape_turn_direction = 1.0 # 1.0 Left, -1.0 Right
        
        self.get_logger().info("MEMORY EXPLORER V2: Anti-Loop Logic Loaded.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Front collision check (Narrower cone to allow tight maneuvering)
        front_indices = np.concatenate((np.arange(0, 10), np.arange(350, 360)))
        if np.min(ranges[front_indices]) < 0.75:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def check_memory(self):
        # Don't check memory if we haven't moved much yet
        if len(self.history) < 50: return False
            
        # Only check against OLD history (older than 5 seconds ago)
        # Assuming 10 frames/sec approx logic update
        old_history = self.history[:-50]
        
        for (past_x, past_y) in old_history:
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            # If we are back within 0.6m of an old spot
            if dist < 0.6:
                return True
        return False

    def find_safest_direction(self):
        if len(self.lidar_ranges) == 0: return 0.0
        left_avg = np.mean(self.lidar_ranges[0:90])
        right_avg = np.mean(self.lidar_ranges[270:360])
        
        # "Gap Finding" Logic
        if left_avg > right_avg: return 0.6 
        else: return -0.6

    def image_callback(self, msg):
        try:
            # 1. Update Memory (Every 5 frames)
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
            
            # --- DECISION TREE ---
            
            # STATE 1: CURRENTLY ESCAPING (High Priority Override)
            if self.escape_timer > 0:
                self.escape_timer -= 1
                
                # ESCAPE MANEUVER: Turn AND Drive
                # This physically forces the robot out of the "Deja Vu" zone
                cmd.twist.linear.x = 0.15
                cmd.twist.angular.z = 0.8 * self.escape_turn_direction
                
                status_text = f"BREAKING LOOP! ({self.escape_timer})"
                color = (255, 0, 255) # Purple
                
            # STATE 2: EMERGENCY BRAKE (Crash Imminent)
            elif self.obstacle_detected:
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = self.find_safest_direction()
                status_text = "OBSTACLE!"
                color = (0, 0, 255) # Red

            # STATE 3: MEMORY CHECK (Deja Vu?)
            elif self.check_memory():
                # We found a loop! Trigger the Escape.
                self.escape_timer = 100  # Commit to escaping for 60 frames (~2-3 sec)
                
                # Randomize the turn to break the pattern
                self.escape_turn_direction = random.choice([-1.0, 1.0])
                
                self.get_logger().warn("LOOP DETECTED! Initiating Random Escape.")

            # STATE 4: NORMAL CRUISE
            else:
                cmd.twist.linear.x = 0.25
                cmd.twist.angular.z = 0.0
                status_text = "PATH CLEAR"

            self.publisher.publish(cmd)
            
            # Visuals
            cv2.putText(frame, f"Nodes: {len(self.history)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            cv2.imshow("Memory Explorer V2", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = MemoryExplorerV2()
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
