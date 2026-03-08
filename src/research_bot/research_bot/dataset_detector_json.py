import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import os
import json
import sys
import datetime
import matplotlib.pyplot as plt

def euler_from_quaternion(q):
    """Convert quaternion (x, y, z, w) to euler yaw angle (radians)"""
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

class LidarDataMiner(Node):
    def __init__(self):
        super().__init__('lidar_data_miner')
        
        # --- CONFIGURATION ---
        self.target_samples = 4000 # Keep low for faster testing!
        self.dataset = []
        
        # --- LOCAL OUTPUT DIRECTORY ---
        self.output_dir = os.path.expanduser("~/ros2_ws/output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_save_path = os.path.join(self.output_dir, f"lidar_dataset_{timestamp}.json")
        self.png_save_path = os.path.join(self.output_dir, f"navigation_map_{timestamp}.png")
        
        # SENSORS & PUBLISHERS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # FAST CONTROL TIMER (10 Hz)
        self.create_timer(0.1, self.control_loop)

        # STATE & NAVIGATION LOGIC
        self.lidar_ranges = []
        self.scan_angles = np.deg2rad(np.arange(360)) # Cache angles 0-359
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.front_dist = 10.0
        self.turn_cooldown = 0
        self.target_turn_direction = 0.0
        self.obstacle_detected = False
        
        # PHYSICS
        self.current_speed = 0.0
        self.acceleration = 0.05 

        self.get_logger().info(f"LiDAR Miner Active! Collecting {self.target_samples} samples and drawing walls...")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = euler_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        self.lidar_ranges = ranges.tolist()
        
        front_indices = np.concatenate((np.arange(0, 30), np.arange(330, 360)))
        self.front_dist = np.min(ranges[front_indices])
        self.obstacle_detected = self.front_dist < 0.55

        if len(self.lidar_ranges) == 360:
            sample = {
                "odom": {"x": self.current_x, "y": self.current_y, "yaw": self.current_yaw},
                "lidar": self.lidar_ranges
            }
            self.dataset.append(sample)
            
            if len(self.dataset) % 100 == 0:
                self.get_logger().info(f"Progress: {len(self.dataset)} / {self.target_samples} scans logged.")

            if len(self.dataset) >= self.target_samples:
                self.save_dataset_and_exit()

    def find_largest_gap(self):
        if len(self.lidar_ranges) == 0: return 0.0
        gaps = [np.mean(self.lidar_ranges[0:90]), np.mean(self.lidar_ranges[90:180]),
                np.mean(self.lidar_ranges[180:270]), np.mean(self.lidar_ranges[270:360])]
        best = gaps.index(max(gaps))
        return 1.0 if best <= 1 else -1.0

    def smooth_velocity(self, cmd, desired):
        if desired < 0: self.current_speed = desired
        else:
            if self.current_speed < desired: self.current_speed += self.acceleration
            elif self.current_speed > desired: self.current_speed -= self.acceleration
        self.current_speed = max(-0.15, min(self.current_speed, 0.15))
        cmd.linear.x = self.current_speed

    def control_loop(self):
        if len(self.dataset) >= self.target_samples:
            return

        cmd = Twist()
        target_speed = 0.20
        turn = 0.0
        
        if self.front_dist < 0.30:
            target_speed = -0.15
        elif self.turn_cooldown > 0:
            self.turn_cooldown -= 1
            target_speed = 0.0
            turn = 0.8 * self.target_turn_direction
        elif self.obstacle_detected:
            self.turn_cooldown = 15
            self.target_turn_direction = self.find_largest_gap()
            target_speed = 0.0
            turn = 0.8 * self.target_turn_direction
        
        cmd.angular.z = float(turn)
        self.smooth_velocity(cmd, target_speed)
        self.publisher.publish(cmd)

    def save_dataset_and_exit(self):
        self.get_logger().info("Target reached! Stopping robot...")
        stop_cmd = Twist()
        self.publisher.publish(stop_cmd)

        # 1. Save JSON
        self.get_logger().info(f"Saving JSON to: {self.json_save_path}")
        with open(self.json_save_path, 'w') as file:
            json.dump(self.dataset, file, indent=4)
            
        # 2. Extract Data and Generate Wall Points
        self.get_logger().info("Calculating global wall coordinates from sensor data...")
        path_x = [sample['odom']['x'] for sample in self.dataset]
        path_y = [sample['odom']['y'] for sample in self.dataset]
        
        wall_x = []
        wall_y = []

        # Only plot every 50th scan to speed up plotting and reduce bloat
        for i in range(0, len(self.dataset), 50):
            sample = self.dataset[i]
            r_x = sample['odom']['x']
            r_y = sample['odom']['y']
            r_yaw = sample['odom']['yaw']
            ranges = np.array(sample['lidar'])
            
            # Use geometry to transform from polar robot coordinates to global x,y
            hit_indices = np.where((ranges > 0.15) & (ranges < 10.0))[0] # Filter out noise and blank space
            for idx in hit_indices:
                r = ranges[idx]
                beam_angle = self.scan_angles[idx] + r_yaw # Combine robot orientation and beam orientation
                
                obs_x = r_x + (r * np.cos(beam_angle))
                obs_y = r_y + (r * np.sin(beam_angle))
                wall_x.append(obs_x)
                wall_y.append(obs_y)

        # 3. Draw Plot
        self.get_logger().info(f"Generating Navigation Map: {self.png_save_path}")
        plt.figure(figsize=(10, 10))
        
        # A. Plot Physical Walls (Black dots)
        plt.scatter(wall_x, wall_y, color='black', s=1, alpha=0.5, label='Physical Walls/Obstacles')

        # B. Plot Path (Blue line)
        plt.plot(path_x, path_y, label='Robot Path', color='blue', linewidth=2, zorder=5)
        
        # C. Plot Start/End markers
        plt.scatter(path_x[0], path_y[0], color='green', label='Start', s=150, zorder=10)
        plt.scatter(path_x[-1], path_y[-1], color='red', label='End', s=150, zorder=10)
        
        # D. Format Plot
        plt.title('Autonomous Navigation with Physical Boundaries')
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Set limits to slightly beyond your 24x24m warehouse walls
        plt.xlim([-15, 15]) 
        plt.ylim([-15, 15])
        plt.axis('equal') 
        
        plt.savefig(self.png_save_path, dpi=300, bbox_inches='tight')
            
        self.get_logger().info("Mission Complete. Shutting down node.")
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = LidarDataMiner()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()