import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_research_bot = get_package_share_directory('research_bot')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    world_file = os.path.join(pkg_research_bot, 'worlds', 'multi_room_warehouse.world')

    # 1. Start Physics Engine with your custom world
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_file}.items()
    )

    # 2. Start 3D Graphics (Will be disabled in Docker)
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    # 3. Publish Robot Sensors/TF
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # 4. Spawn the physical robot into the world
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'waffle_pi',
            '-file', os.path.join(pkg_tb3_gazebo, 'models', 'turtlebot3_waffle_pi', 'model.sdf'),
            '-x', '0.0', '-y', '0.0', '-z', '0.01'
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    # ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_entity)

    return ld
