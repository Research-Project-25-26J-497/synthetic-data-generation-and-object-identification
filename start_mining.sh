#!/bin/bash

# 1. Source the ROS 2 environment
source /opt/ros/humble/setup.bash
source /app/ros2_ws/install/setup.bash

# 2. CRITICAL FIX: Block Gazebo from hanging on dead online servers
export GAZEBO_MODEL_DATABASE_URI=""

# 3. Setup Software Rendering & TurtleBot3 Models
export LIBGL_ALWAYS_SOFTWARE=1
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models

# 4. Launch the simulation securely wrapped in a managed virtual screen
echo "Starting Gazebo Physics Engine..."
xvfb-run -a -s "-screen 0 1024x768x24" ros2 launch research_bot sim.launch.py &

# 5. Give the server time to spawn the robot and start the ROS bridge
echo "Waiting 15 seconds for physics server to stabilize..."
sleep 15

# 6. Run your Python data collector
echo "Starting the LiDAR JSON Miner..."
ros2 run research_bot dataset_generator