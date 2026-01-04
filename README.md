# Autonomous Self-Learning Exploration Robot

This ROS 2 package implements a robot capable of autonomous navigation, exploration, and self-supervised online learning.

## 🚀 Features
1. **Sensor Fusion Navigation:** Uses LIDAR for obstacle avoidance and Camera for object detection.
2. **Memory-Based Loop Breaking:** Detects "Dejà vu" (repetitive paths) and forces the robot to explore new areas.
3. **Self-Supervised Learning:** The robot autonomously collects data, pseudo-labels it, and retrains its YOLOv8 neural network in real-time.

## 🛠️ Installation
1. Clone this repo into your ROS 2 workspace src folder:
   ```bash
   cd ~/ros2_ws/src
   git clone <your-repo-link>