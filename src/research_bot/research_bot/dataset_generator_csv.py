import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import os
import csv
import sys

class LidarDataMiner(Node):
    def __init__(self):
        super().__init__('lidar_data_miner')
        
        # --- CONFIGURATION ---
        self.target_samples = 2000  # How many rows of data to collect before stopping
        self.save_path = os.path.expanduser("~/ros2_ws/lidar_dataset.csv")
        self.dataset = []
        
        # SENSORS & PUBLISHERS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # FAST CONTROL TIMER (10 Hz)
        self.create_timer(0.1, self.control_loop)

        # STATE
        self.lidar_ranges = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.front_dist = 10.0
        
        # NAVIGATION LOGIC
        self.turn_cooldown = 0
        self.target_turn_direction = 0.0
        self.obstacle_detected = False
        
        # PHYSICS
        self.current_speed = 0.0
        self.acceleration = 0.05 

        self.get_logger().info(f"LiDAR Miner Active! Collecting {self.target_samples} samples...")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        # Clean the incoming LiDAR data (replace 'inf' with 10.0 meters)
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        self.lidar_ranges = ranges.tolist()
        
        # Navigation logic (look at the front 60 degrees)
        front_indices = np.concatenate((np.arange(0, 30), np.arange(330, 360)))
        self.front_dist = np.min(ranges[front_indices])
        self.obstacle_detected = self.front_dist < 0.55

        # --- DATA COLLECTION ---
        # Only collect data if we have a valid scan
        if len(self.lidar_ranges) == 360:
            row = self.lidar_ranges + [self.current_x, self.current_y]
            self.dataset.append(row)
            
            # Print progress every 100 samples
            if len(self.dataset) % 100 == 0:
                self.get_logger().info(f"Progress: {len(self.dataset)} / {self.target_samples} scans logged.")

            # Stop and save when we hit the target
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
        # If we finished collecting data, stop moving
        if len(self.dataset) >= self.target_samples:
            return

        cmd = Twist()

        target_speed = 0.20 # Faster exploration
        turn = 0.0
        
        if self.front_dist < 0.30:
            target_speed = -0.15 # Reverse if too close
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
        self.get_logger().info(f"Target reached! Saving {self.target_samples} rows to CSV...")
        
        # Stop the robot
        stop_cmd = Twist()
        self.publisher.publish(stop_cmd)

        # Write to CSV
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Create Headers (Lidar_0 to Lidar_359, then Odom_X, Odom_Y)
            headers = [f"Lidar_{i}" for i in range(360)] + ["Odom_X", "Odom_Y"]
            writer.writerow(headers)
            
            # Write all data rows
            writer.writerows(self.dataset)
            
        self.get_logger().info(f"Dataset saved to: {self.save_path}")
        self.get_logger().info("Mission Complete. Shutting down node.")
        
        # In the Docker Phase, the S3 Boto3 upload trigger will go right here.
        
        sys.exit(0) # Tells Docker the process finished successfully


def main(args=None):
    rclpy.init(args=args)
    node = LidarDataMiner()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass # Expected exit when dataset is done
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()