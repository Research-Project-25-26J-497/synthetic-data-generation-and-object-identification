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
import time
import os
import shutil
import yaml

class SelfLearningBot(Node):
    def __init__(self):
        super().__init__('self_learning_bot')
        
        # --- CONFIGURATION ---
        self.batch_size = 30  # Train after every 30 new images (Small for demo)
        self.epochs = 5       # Quick "Study Session" (prevent freezing for too long)
        self.base_path = os.path.expanduser("~/ros2_ws/auto_dataset")
        
        # Setup Directories
        self.train_img_dir = os.path.join(self.base_path, "train/images")
        self.train_lbl_dir = os.path.join(self.base_path, "train/labels")
        self.setup_directories()
        
        # Load Initial Model (Standard or previously trained)
        self.model_path = 'yolov8n.pt'
        self.model = YOLO(self.model_path)
        self.generation = 0 # Track how many times we've trained
        
        # SENSORS & PUBS
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        # STATE VARIABLES
        self.state = "CRUISE" # CRUISE, TRAINING
        self.lidar_ranges = []
        self.history = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.frame_counter = 0
        self.front_dist = 10.0
        
        # DATA COLLECTION STATE
        self.collected_count = 0
        self.last_save_time = 0
        self.turn_cooldown = 0
        self.target_turn_direction = 0.0
        self.obstacle_detected = False
        self.memory_cooldown = 0
        
        # PHYSICS
        self.current_speed = 0.0
        self.acceleration = 0.005

        self.get_logger().info(f"SELF-LEARNING BOT: Generation {self.generation} Active.")

    def setup_directories(self):
        # Reset or Create Folder Structure
        if not os.path.exists(self.train_img_dir): os.makedirs(self.train_img_dir)
        if not os.path.exists(self.train_lbl_dir): os.makedirs(self.train_lbl_dir)
        
        # Create data.yaml required by YOLO
        yaml_content = {
            'path': self.base_path,
            'train': 'train/images',
            'val': 'train/images', # Just use train data for validation in this hack
            'names': {0: 'object'} # Generic class name
        }
        with open(os.path.join(self.base_path, 'data.yaml'), 'w') as f:
            yaml.dump(yaml_content, f)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        self.lidar_ranges = ranges
        front_indices = np.concatenate((np.arange(0, 30), np.arange(330, 360)))
        self.front_dist = np.min(ranges[front_indices])
        self.obstacle_detected = self.front_dist < 0.55

    def find_largest_gap(self):
        if len(self.lidar_ranges) == 0: return 0.0
        q1 = np.mean(self.lidar_ranges[0:90])
        q2 = np.mean(self.lidar_ranges[90:180])
        q3 = np.mean(self.lidar_ranges[180:270])
        q4 = np.mean(self.lidar_ranges[270:360])
        gaps = [q1, q2, q3, q4]
        best = gaps.index(max(gaps))
        return 1.0 if best <= 1 else -1.0

    def check_memory(self):
        if self.memory_cooldown > 0:
            self.memory_cooldown -= 1
            return False
        if len(self.history) < 200: return False
        old = self.history[:-150]
        for (px, py) in old:
            if math.sqrt((self.current_x - px)**2 + (self.current_y - py)**2) < 0.8:
                return True
        return False

    def train_model(self):
        self.get_logger().warn("STARTING TRAINING SESSION... (Robot will pause)")
        
        try:
            # Run YOLO Training
            # We use 'exist_ok=True' to overwrite previous runs or continue
            results = self.model.train(
                data=os.path.join(self.base_path, 'data.yaml'),
                epochs=self.epochs,
                imgsz=640,
                batch=4, # Small batch for weak laptops
                project=os.path.join(self.base_path, 'runs'),
                name=f'gen_{self.generation}',
                verbose=True
            )
            
            # Find the new weights
            new_weights = os.path.join(self.base_path, f'runs/gen_{self.generation}/weights/best.pt')
            
            if os.path.exists(new_weights):
                self.get_logger().info("TRAINING COMPLETE! Hot-swapping model...")
                self.model = YOLO(new_weights) # Load the new brain
                self.generation += 1
                self.collected_count = 0 # Reset counter for next batch
            else:
                self.get_logger().error("Training finished but weights not found.")
                
        except Exception as e:
            self.get_logger().error(f"TRAINING FAILED: {e}")
        
        self.get_logger().info("RESUMING EXPLORATION.")
        self.state = "CRUISE"

    def save_training_data(self, frame, results):
        # Save Image
        timestamp = int(time.time() * 1000)
        img_name = f"{timestamp}.jpg"
        img_path = os.path.join(self.train_img_dir, img_name)
        cv2.imwrite(img_path, frame)
        
        # Save Label (YOLO Format: class x_center y_center width height)
        lbl_path = os.path.join(self.train_lbl_dir, f"{timestamp}.txt")
        
        with open(lbl_path, 'w') as f:
            for box in results[0].boxes:
                # We auto-label everything as class '0' (Generic Object)
                # to enable simple obstacle detection training
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"0 {x} {y} {w} {h}\n")
        
        self.collected_count += 1

    def smooth_velocity(self, cmd, desired):
        if desired < 0: self.current_speed = desired
        else:
            if self.current_speed < desired: self.current_speed += self.acceleration
            elif self.current_speed > desired: self.current_speed -= self.acceleration
        self.current_speed = max(-0.15, min(self.current_speed, 0.15))
        cmd.twist.linear.x = self.current_speed

    def image_callback(self, msg):
        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # --- LOGIC CONTROL ---
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            if self.state == "TRAINING":
                # Do nothing, just wait
                return

            # RUN INFERENCE (With current brain)
            results = self.model(raw_frame, verbose=False, conf=0.25)
            
            # AUTO-LABEL & COLLECT
            if len(results[0].boxes) > 0 and (time.time() - self.last_save_time > 1.0):
                self.save_training_data(raw_frame, results)
                self.last_save_time = time.time()
                
                # TRIGGER TRAINING?
                if self.collected_count >= self.batch_size:
                    self.target_speed = 0.0
                    self.smooth_velocity(cmd, 0.0)
                    self.publisher.publish(cmd) # Stop robot first
                    self.state = "TRAINING"
                    self.train_model() # This will block!
                    return

            # --- NAVIGATION STACK (V5 Logic) ---
            
            # 1. Update History
            self.frame_counter += 1
            if self.frame_counter % 10 == 0:
                self.history.append((self.current_x, self.current_y))

            # 2. State Machine
            target_speed = 0.15
            turn = 0.0
            status = "CRUISE"
            color = (0, 255, 0)
            
            if self.front_dist < 0.30:
                target_speed = -0.15
                status = "TOO CLOSE!"
                color = (0, 0, 255)
            elif self.turn_cooldown > 0:
                self.turn_cooldown -= 1
                target_speed = 0.0
                turn = 0.6 * self.target_turn_direction
                status = "LOCKED TURN"
                color = (255, 165, 0)
            elif self.obstacle_detected:
                self.turn_cooldown = 15
                self.target_turn_direction = self.find_largest_gap()
                target_speed = 0.0
                turn = 0.6 * self.target_turn_direction
                status = "AVOIDING"
                color = (0, 255, 255)
            elif self.check_memory():
                self.turn_cooldown = 30 # Simple escape
                self.target_turn_direction = self.find_largest_gap()
                status = "LOOP ESCAPE"
                color = (255, 0, 255)
            
            # Apply Drive
            self.target_speed = target_speed
            cmd.twist.angular.z = float(turn)
            self.smooth_velocity(cmd, target_speed)
            self.publisher.publish(cmd)
            
            # --- VISUALIZATION ---
            frame = cv2.resize(raw_frame, (0, 0), fx=1.5, fy=1.5)
            
            # Draw boxes from current brain
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Scale coordinates to match the 1.5x resized frame
                x1, y1, x2, y2 = x1*1.5, y1*1.5, x2*1.5, y2*1.5
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Stats
            cv2.putText(frame, f"Gen: {self.generation} | Data: {self.collected_count}/{self.batch_size}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Self-Learning Bot", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SelfLearningBot()
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
