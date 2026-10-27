"""The lab: sim_launch.py on a lab duct map with the real lab car's sensors.

  ros2 launch f1tenth_gym_gazebo lab_launch.py                    # lab_00 (ICRA shape)
  ros2 launch f1tenth_gym_gazebo lab_launch.py map_name:=lab_05   # any lab_00..lab_20

Real car: Intel RealSense D435i colour stream (16:9, 69 x 42 deg, 30 Hz) just above the
2D Hokuyo, level; no 3D lidar. Maps from lane_assist/tools/maps/make_lab_maps.py.
Same values as sim_args in lane_assist_params.yaml (lab_collect): change both together.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sim = os.path.join(get_package_share_directory('f1tenth_gym_gazebo'), 'launch', 'sim_launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='lab_00', description='lab_00 .. lab_20'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim), launch_arguments={
            'map_name': LaunchConfiguration('map_name'),
            'rviz': LaunchConfiguration('rviz'),
            'gui': LaunchConfiguration('gui'),
            'camera_mount_z': '0.2025',
            'camera_pitch': '0.0',
            'camera_image_width': '640',
            'camera_image_height': '360',
            'camera_rate': '30',
            'enable_lidar3d': 'false',
        }.items()),
    ])
