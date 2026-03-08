# 1. Use the official ROS 2 Humble Desktop image (includes Gazebo)
FROM osrf/ros:humble-desktop

# 2. Set up environment variables
ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=waffle_pi

# 3. Install Python pip, our fake screen tool (xvfb), and TurtleBot3 packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    xvfb \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app/ros2_ws

# 5. Copy the requirements and install Python tools (Matplotlib, Boto3)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 6. Copy your local workspace into the container (ignores build/install folders)
COPY . .

# 7. Compile the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# 8. Set the entry point to our startup script
ENTRYPOINT ["/app/ros2_ws/start_mining.sh"]
