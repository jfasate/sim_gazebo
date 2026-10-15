import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'f1tenth_gym_gazebo'

# Collect all model files recursively
model_data_files = []
for dirpath, dirnames, filenames in os.walk('models'):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        install_dir = os.path.join('share', package_name, dirpath)
        model_data_files.append((install_dir, [filepath]))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),

        # URDF / Xacro
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),

        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),

        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),

        # RViz layouts (sim_launch resolves rviz/f1tenth_gazebo.rviz)
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),

        # 2D occupancy-grid maps (map_server format)
        (os.path.join('share', package_name, 'maps'),
            glob('maps/*')),

    ] + model_data_files,

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jayesh Fasate',
    maintainer_email='jfasate@purdue.edu',
    description='Gazebo Classic simulation for F1Tenth VLM-MPC project',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dynamics_plant = f1tenth_gym_gazebo.dynamics_plant:main',
            'sensor_delay = f1tenth_gym_gazebo.sensor_delay:main',
            'sysid_excite = f1tenth_gym_gazebo.sysid_excite:main',
            'sysid_fit = f1tenth_gym_gazebo.sysid_fit:main',
            'world_to_map = f1tenth_gym_gazebo.world_to_map:main',
            'map_publisher = f1tenth_gym_gazebo.map_publisher:main',
            'ackermann_to_twist = f1tenth_gym_gazebo.ackermann_to_twist:main',
            'pose_reset = f1tenth_gym_gazebo.pose_reset:main',
            'teleop_keyboard = f1tenth_gym_gazebo.teleop_keyboard:main',
            'teleop_bridge = f1tenth_gym_gazebo.teleop_bridge:main',
            'save_map = f1tenth_gym_gazebo.save_map:main',
            'dynamic_obstacles = f1tenth_gym_gazebo.dynamic_obstacles:main',
        ],
    },
)
