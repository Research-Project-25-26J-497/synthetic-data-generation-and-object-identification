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

class VisualMemoryExplorerV3(Node):
    def __init__(self):
        super().__init__('visual_memory_explorer_v3')
        
        self.model = YOLO('yolov8n.pt')
        
        # SENSORS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # NAVIGATION STATE
        self.lidar_ranges = []
        self.obstacle_detected = False
        self.history = [] 
        self.current_x = 0.0
        self.current_y = 0.0
        self.frame_counter = 0
        self.front_dist = 10.0
        
        # STATE MACHINE
        self.state = "CRUISE"
        self.state_timer = 0
        self.target_turn_direction = 0.0 
        self.memory_cooldown = 0
        
        # --- PHYSICS CONTROL (ANTI-FLIP) ---
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.acceleration = 0.002  # ULTRA LOW for stability (was 0.01)

        self.get_logger().info("V3 EXPLORER: Big Screen Mode & Anti-Flip Physics.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Front collision check
        front_indices = np.concatenate((np.arange(0, 15), np.arange(345, 360)))
        self.front_dist = np.min(ranges[front_indices])
        
        # Increased safety distance to prevent wall climbing
        if self.front_dist < 0.65:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def find_largest_gap(self):
        if len(self.lidar_ranges) == 0: return 0.0
        q1 = np.mean(self.lidar_ranges[0:90])    # Front Left
        q2 = np.mean(self.lidar_ranges[90:180])  # Back Left
        q3 = np.mean(self.lidar_ranges[180:270]) # Back Right
        q4 = np.mean(self.lidar_ranges[270:360]) # Front Right
        
        gaps = [q1, q2, q3, q4]
        best_quadrant = gaps.index(max(gaps))
        
        if best_quadrant == 0 or best_quadrant == 1: return 1.0 
        else: return -1.0

    def check_memory(self):
        if self.memory_cooldown > 0:
            self.memory_cooldown -= 1
            return False

        if len(self.history) < 50: return False
        
        limit = min(len(self.history), 300)
        old_history = self.history[:-limit]
        
        for (past_x, past_y) in old_history:
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            if dist < 1.5: return True
        return False

    def draw_minimap(self, frame):
        # Increased Map Size for visibility
        map_size = 300 
        scale = 20 
        center = map_size // 2
        minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # Draw Trail
        for (hx, hy) in self.history:
            mx = int(center + (hx * scale))
            my = int(center - (hy * scale))
            if 0 <= mx < map_size and 0 <= my < map_size:
                cv2.circle(minimap, (mx, my), 2, (200, 200, 200), -1)

        # Draw Robot
        rx = int(center + (self.current_x * scale))
        ry = int(center - (self.current_y * scale))
        if 0 <= rx < map_size and 0 <= ry < map_size:
            cv2.circle(minimap, (rx, ry), 6, (0, 0, 255), -1)

        # Overlay
        frame[0:map_size, 0:map_size] = minimap
        cv2.rectangle(frame, (0,0), (map_size, map_size), (0, 255, 0), 2)
        cv2.putText(frame, "SLAM MAP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def smooth_velocity_control(self, cmd, desired_linear):
        """ Prevents wheelies by ramping speed very slowly """
        if self.current_speed < desired_linear:
            self.current_speed += self.acceleration
        elif self.current_speed > desired_linear:
            self.current_speed -= self.acceleration
            
        # Capped Max Speed to 0.15 for safety
        self.current_speed = max(0.0, min(self.current_speed, 0.15))
        cmd.twist.linear.x = self.current_speed

    def image_callback(self, msg):
        try:
            self.frame_counter += 1
            if self.frame_counter % 10 == 0: 
                self.history.append((self.current_x, self.current_y))
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # --- BIG SCREEN FIX ---
            # Instead of 0.5 (shrinking), we use 1.5 (Zooming In)
            frame = cv2.resize(raw_frame, (0, 0), fx=1.5, fy=1.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "CRUISING"
            color = (0, 255, 0)
            
            # --- STATE MACHINE ---
            
            # 1. EMERGENCY STOP
            if self.obstacle_detected:
                self.state = "OBSTACLE"
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.5 * self.find_largest_gap()
                status_text = "TOO CLOSE!"
                color = (0, 0, 255)
            
            # 2. STOP AND THINK
            elif self.state == "STOP_AND_THINK":
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.0
                self.state_timer -= 1
                status_text = "CALCULATING..."
                color = (0, 255, 255)
                
                if self.state_timer <= 0:
                    self.target_turn_direction = self.find_largest_gap()
                    self.state = "PIVOT_ALIGN"
                    self.state_timer = 50 

            # 3. PIVOT ALIGN
            elif self.state == "PIVOT_ALIGN":
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.6 * self.target_turn_direction
                self.state_timer -= 1
                status_text = "ALIGNING..."
                color = (255, 0, 255)
                
                if self.state_timer <= 0:
                    self.state = "ESCAPE_DRIVE"
                    self.state_timer = 120 

            # 4. ESCAPE DRIVE
            elif self.state == "ESCAPE_DRIVE":
                self.target_speed = 0.15
                cmd.twist.angular.z = 0.0
                self.state_timer -= 1
                status_text = "ESCAPING LOOP"
                color = (255, 0, 255)
                
                if self.state_timer <= 0:
                    self.state = "CRUISE"
                    self.memory_cooldown = 200

            # 5. CRUISE
            else:
                self.state = "CRUISE"
                if self.check_memory():
                    self.state = "STOP_AND_THINK"
                    self.state_timer = 30
                    self.target_speed = 0.0
                    self.get_logger().warn("Loop Detected!")
                else:
                    self.target_speed = 0.15
                    cmd.twist.angular.z = 0.0
                    status_text = "PATH CLEAR"

            # Apply Anti-Flip Physics
            self.smooth_velocity_control(cmd, self.target_speed)
            
            self.publisher.publish(cmd)
            self.draw_minimap(frame)
            
            # UI Text (Positioned for the larger window)
            cv2.putText(frame, status_text, (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Power Bar
            speed_bar = int(self.current_speed * 1500)
            cv2.rectangle(frame, (320, 80), (320 + speed_bar, 110), (255, 255, 255), -1)
            cv2.putText(frame, "Engine Power", (320, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("V3 Explorer (Big Screen)", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VisualMemoryExplorerV3()
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
