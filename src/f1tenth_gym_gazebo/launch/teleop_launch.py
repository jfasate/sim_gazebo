"""
Launch file for F1Tenth teleop keyboard control.

NOTE: Because the keyboard node needs direct terminal access (termios/stdin),
      use 'ros2 run' instead of 'ros2 launch':

  Terminal 1:  ros2 launch f1tenth_gym_gazebo sim_launch.py
  Terminal 2:  ros2 run f1tenth_gym_gazebo teleop_keyboard

The teleop_keyboard node publishes AckermannDriveStamped directly to /drive,
which the ackermann_to_twist bridge (in sim_launch) converts to /cmd_vel
for Gazebo.
"""

from launch import LaunchDescription


def generate_launch_description():
    # This launch file is intentionally empty.
    # Use: ros2 run f1tenth_gym_gazebo teleop_keyboard
    return LaunchDescription([])
