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

class VisualMemoryExplorerV5(Node):
    def __init__(self):
        super().__init__('visual_memory_explorer_v5')
        
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
        
        # NEW: TURN COMMITMENT (Prevents Jitter)
        self.turn_cooldown = 0 
        
        # PHYSICS CONTROL
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.acceleration = 0.005 

        self.get_logger().info("V5 EXPLORER: Turn Commitment & Closer Proximity.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Front collision check (WIDER now to catch corners)
        # 30 degrees Left + 30 degrees Right
        front_indices = np.concatenate((np.arange(0, 30), np.arange(330, 360)))
        self.front_dist = np.min(ranges[front_indices])
        
        # Standard Avoidance Threshold (Lowered to 0.55m)
        if self.front_dist < 0.55:
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
        map_size = 300 
        scale = 20 
        center = map_size // 2
        minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        for (hx, hy) in self.history:
            mx = int(center + (hx * scale))
            my = int(center - (hy * scale))
            if 0 <= mx < map_size and 0 <= my < map_size:
                cv2.circle(minimap, (mx, my), 2, (200, 200, 200), -1)

        rx = int(center + (self.current_x * scale))
        ry = int(center - (self.current_y * scale))
        if 0 <= rx < map_size and 0 <= ry < map_size:
            cv2.circle(minimap, (rx, ry), 6, (0, 0, 255), -1)

        frame[0:map_size, 0:map_size] = minimap
        cv2.rectangle(frame, (0,0), (map_size, map_size), (0, 255, 0), 2)
        cv2.putText(frame, "SLAM MAP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def smooth_velocity_control(self, cmd, desired_linear):
        if desired_linear < 0:
            self.current_speed = desired_linear 
        else:
            if self.current_speed < desired_linear:
                self.current_speed += self.acceleration
            elif self.current_speed > desired_linear:
                self.current_speed -= self.acceleration
        
        self.current_speed = max(-0.15, min(self.current_speed, 0.15))
        cmd.twist.linear.x = self.current_speed

    def image_callback(self, msg):
        try:
            self.frame_counter += 1
            if self.frame_counter % 10 == 0: 
                self.history.append((self.current_x, self.current_y))
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=1.5, fy=1.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "CRUISING"
            color = (0, 255, 0)
            
            # --- PRIORITY 1: CRITICAL REVERSE ---
            if self.state == "REVERSE_RECOVERY":
                self.state_timer -= 1
                self.target_speed = -0.15 # Back up faster
                cmd.twist.angular.z = 0.0 
                status_text = "REVERSING..."
                color = (0, 0, 255)
                
                if self.state_timer <= 0:
                    self.state = "PIVOT_ALIGN"
                    self.target_turn_direction = self.find_largest_gap()
                    self.state_timer = 40

            # --- PRIORITY 2: PROXIMITY CHECK (Lowered to 0.30m) ---
            # Now we use 'front_dist' which is wider (catch corners)
            elif self.front_dist < 0.30:
                self.state = "REVERSE_RECOVERY"
                self.state_timer = 30 # 1 second reverse
                self.target_speed = 0.0 
                status_text = "TOO CLOSE! BACKING UP"
                color = (0, 0, 255)

            # --- PRIORITY 3: COMMITTED TURN (New!) ---
            # If we started turning recently, KEEP TURNING. Don't jitter.
            elif self.turn_cooldown > 0:
                self.turn_cooldown -= 1
                self.target_speed = 0.0
                # Keep turning in the direction we decided
                cmd.twist.angular.z = 0.6 * self.target_turn_direction
                status_text = "AVOIDING (LOCKED)"
                color = (255, 165, 0) # Orange

            # --- PRIORITY 4: OBSTACLE AVOIDANCE START ---
            elif self.obstacle_detected: # Between 0.30m and 0.55m
                # Start a new committed turn
                self.turn_cooldown = 15 # Commit for 0.5 seconds minimum
                self.target_turn_direction = self.find_largest_gap()
                
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.6 * self.target_turn_direction
                status_text = "AVOIDING"
                color = (0, 255, 255) 

            # --- PRIORITY 5: THINK/ALIGN/ESCAPE ---
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

            elif self.state == "PIVOT_ALIGN":
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.6 * self.target_turn_direction
                self.state_timer -= 1
                status_text = "ALIGNING..."
                color = (255, 0, 255)
                if self.state_timer <= 0:
                    self.state = "ESCAPE_DRIVE"
                    self.state_timer = 120 

            elif self.state == "ESCAPE_DRIVE":
                self.target_speed = 0.15
                cmd.twist.angular.z = 0.0
                self.state_timer -= 1
                status_text = "ESCAPING LOOP"
                color = (255, 0, 255)
                if self.state_timer <= 0:
                    self.state = "CRUISE"
                    self.memory_cooldown = 200

            # --- PRIORITY 6: CRUISE ---
            else:
                self.state = "CRUISE"
                if self.check_memory():
                    self.state = "STOP_AND_THINK"
                    self.state_timer = 30
                    self.target_speed = 0.0
                else:
                    self.target_speed = 0.15
                    cmd.twist.angular.z = 0.0
                    status_text = "PATH CLEAR"

            self.smooth_velocity_control(cmd, self.target_speed)
            
            self.publisher.publish(cmd)
            self.draw_minimap(frame)
            
            cv2.putText(frame, status_text, (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            speed_bar = int(abs(self.current_speed) * 1500)
            bar_color = (0, 0, 255) if self.current_speed < 0 else (255, 255, 255)
            cv2.rectangle(frame, (320, 80), (320 + speed_bar, 110), bar_color, -1)
            cv2.putText(frame, "Engine Power", (320, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("V5 Explorer (Tuned)", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VisualMemoryExplorerV5()
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
