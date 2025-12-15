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

class VisualMemoryExplorerV2(Node):
    def __init__(self):
        super().__init__('visual_memory_explorer_v2')
        
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
        
        # --- NEW: STATE MACHINE VARIABLES ---
        # States: "CRUISE", "STOP_AND_THINK", "PIVOT_ALIGN", "ESCAPE_DRIVE"
        self.state = "CRUISE"
        self.state_timer = 0
        self.target_turn_direction = 0.0 
        self.memory_cooldown = 0
        
        # --- NEW: PHYSICS CONTROL ---
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.acceleration = 0.01  # How fast we change speed (Lower = Smoother)

        self.get_logger().info("V2 EXPLORER: Soft-Start Physics & Pivot Turns Active.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges

        # Front collision check (Very close)
        front_indices = np.concatenate((np.arange(0, 15), np.arange(345, 360)))
        self.front_dist = np.min(ranges[front_indices])
        
        if self.front_dist < 0.5:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def find_largest_gap(self):
        if len(self.lidar_ranges) == 0: return 0.0
        
        # Analyze Quadrants
        q1 = np.mean(self.lidar_ranges[0:90])    # Front Left
        q2 = np.mean(self.lidar_ranges[90:180])  # Back Left
        q3 = np.mean(self.lidar_ranges[180:270]) # Back Right
        q4 = np.mean(self.lidar_ranges[270:360]) # Front Right
        
        gaps = [q1, q2, q3, q4]
        best_quadrant = gaps.index(max(gaps))
        
        # Return turn direction (+1 Left, -1 Right)
        if best_quadrant == 0 or best_quadrant == 1: return 1.0 
        else: return -1.0

    def check_memory(self):
        if self.memory_cooldown > 0:
            self.memory_cooldown -= 1
            return False

        if len(self.history) < 50: return False
        
        # Trigger radius: 1.5 meters (Good balance)
        limit = min(len(self.history), 300)
        old_history = self.history[:-limit]
        
        for (past_x, past_y) in old_history:
            dist = math.sqrt((self.current_x - past_x)**2 + (self.current_y - past_y)**2)
            if dist < 1.5: return True
        return False

    def draw_minimap(self, frame):
        map_size = 200 
        scale = 15 
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
            cv2.circle(minimap, (rx, ry), 5, (0, 0, 255), -1)

        frame[0:map_size, 0:map_size] = minimap
        cv2.rectangle(frame, (0,0), (map_size, map_size), (0, 255, 0), 2)

    def smooth_velocity_control(self, cmd, desired_linear):
        """ Ramps speed up/down slowly to prevent wheelies """
        if self.current_speed < desired_linear:
            self.current_speed += self.acceleration
        elif self.current_speed > desired_linear:
            self.current_speed -= self.acceleration
            
        # Clamp value
        self.current_speed = max(0.0, min(self.current_speed, 0.2)) # Max speed capped at 0.2
        cmd.twist.linear.x = self.current_speed

    def image_callback(self, msg):
        try:
            self.frame_counter += 1
            if self.frame_counter % 20 == 0: 
                self.history.append((self.current_x, self.current_y))
            
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            status_text = "CRUISING"
            color = (0, 255, 0)
            
            # --- THE "STOP-THINK-ACT" STATE MACHINE ---
            
            # 1. EMERGENCY COLLISION OVERRIDE (Always Active)
            if self.obstacle_detected:
                self.state = "OBSTACLE" # Force state change
                self.target_speed = 0.0 # Stop immediately
                cmd.twist.angular.z = 0.5 * self.find_largest_gap() # Rotate away
                status_text = "TOO CLOSE!"
                color = (0, 0, 255)
            
            # 2. STATE: STOP AND THINK (Calculated Pause)
            elif self.state == "STOP_AND_THINK":
                self.target_speed = 0.0
                cmd.twist.angular.z = 0.0
                
                # Wait for 30 frames (1 second) to simulate thinking
                self.state_timer -= 1
                status_text = "CALCULATING..."
                color = (0, 255, 255) # Yellow
                
                if self.state_timer <= 0:
                    # Decision made: Switch to Pivot
                    self.target_turn_direction = self.find_largest_gap()
                    self.state = "PIVOT_ALIGN"
                    self.state_timer = 40 # Rotate for ~1.5 seconds

            # 3. STATE: PIVOT ALIGN (Turn in Place)
            elif self.state == "PIVOT_ALIGN":
                self.target_speed = 0.0 # DO NOT MOVE FORWARD
                cmd.twist.angular.z = 0.6 * self.target_turn_direction
                
                self.state_timer -= 1
                status_text = "ALIGNING..."
                color = (255, 0, 255) # Purple
                
                if self.state_timer <= 0:
                    # Aligned: Switch to Drive
                    self.state = "ESCAPE_DRIVE"
                    self.state_timer = 100 # Drive for ~3 seconds

            # 4. STATE: ESCAPE DRIVE (Move Forward Stable)
            elif self.state == "ESCAPE_DRIVE":
                self.target_speed = 0.15 # Slow, steady speed
                cmd.twist.angular.z = 0.0 # Drive straight
                
                self.state_timer -= 1
                status_text = "ESCAPING LOOP"
                color = (255, 0, 255)
                
                if self.state_timer <= 0:
                    # Escape complete
                    self.state = "CRUISE"
                    self.memory_cooldown = 200 # Don't check memory for a while

            # 5. STATE: NORMAL CRUISE
            else:
                self.state = "CRUISE"
                # Check for Deja Vu?
                if self.check_memory():
                    # TRIGGER THE PROCESS
                    self.state = "STOP_AND_THINK"
                    self.state_timer = 30 # Pause duration
                    self.target_speed = 0.0
                    self.get_logger().warn("Loop Detected! Pausing to calculate...")
                
                else:
                    # Standard behavior
                    self.target_speed = 0.15
                    cmd.twist.angular.z = 0.0
                    status_text = "PATH CLEAR"

            # APPLY PHYSICS (Smooth Acceleration)
            self.smooth_velocity_control(cmd, self.target_speed)
            
            # VISUALIZATIONS
            self.publisher.publish(cmd)
            self.draw_minimap(frame)
            cv2.putText(frame, status_text, (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            # Show "Speed" bar to prove smoothness
            speed_bar = int(self.current_speed * 1000)
            cv2.rectangle(frame, (220, 80), (220 + speed_bar, 100), (255, 255, 255), -1)
            cv2.putText(frame, "Power", (220, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("V2 Explorer", frame)
            cv2.waitKey(1)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VisualMemoryExplorerV2()
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
