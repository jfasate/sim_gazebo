"""Launch the F1TENTH Gazebo simulation on any generated map.

  colcon build --packages-select f1tenth_gym_gazebo
  source install/setup.bash
  ros2 launch f1tenth_gym_gazebo sim_launch.py

Defaults live in config/sim.yaml (map_name, rviz/gui, sensor flags) so the
common case needs no arguments. Every one is still overridable on the CLI:

  ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:=oval_loop
  ros2 launch f1tenth_gym_gazebo sim_launch.py enable_camera_d435:=false

Brings up Gazebo, the robot spawn, dynamics_plant, the /map publisher and
optional RViz. It intentionally does not start MPC, neural_generator, teleop,
dynamic_obstacles or data collection — run those separately.
"""

import os
import subprocess
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            LogInfo, OpaqueFunction, SetEnvironmentVariable,
                            TimerAction)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Matches the generator's hardcoded workspace base. Runtime should use the
# installed package after colcon build; source is only a fallback for development.
SRC_PKG = '/home/jayesh/sim_gazebo/src/f1tenth_gym_gazebo'
_BASE_MAPS = ['test_worldv5',
              'straight_corridor', 's_curve', 'oval_loop', 'superspeedway',
              't_junction', 'obstacle_corridor',
              # 15 additional safe-navigation maps (generate_safe_gazebo_maps.py)
              'gentle_chicane', 'double_s_curve', 'wide_junction', 'narrow_gate',
              'offset_obstacles', 'parking_lot_loop', 'hairpin_safe',
              'figure_eight_soft', 'warehouse_aisles', 'slalom_corridor',
              'fork_choice', 'roundabout_small', 'maze_corridor_easy',
              'construction_zone', 'campus_loop']

# Fallbacks if config/sim.yaml is missing or unreadable.
_FALLBACK_DEFAULTS = {
    'map_name': 'superspeedway', 'rviz': True, 'gui': True, 'spawn_car': True,
    'require_clean_runtime': True, 'enable_lidar2d': True, 'enable_lidar3d': True,
    'enable_camera_d435': True, 'enable_imu': True,
    'camera_mount_z': 0.40, 'camera_pitch': 0.0463,
    'camera_image_width': 640, 'camera_image_height': 480, 'camera_rate': 15,
}


def _filtered_path():
    """Prefer system ROS/Python tools over Conda/Snap wrappers."""
    ros_distro = os.environ.get('ROS_DISTRO', 'humble')
    preferred = [
        os.path.join('/opt/ros', ros_distro, 'bin'),
        '/usr/local/sbin', '/usr/local/bin',
        '/usr/sbin', '/usr/bin', '/sbin', '/bin',
    ]
    blocked = ('/miniconda', '/anaconda', '/snap/bin')
    current = os.environ.get('PATH', '').split(os.pathsep)
    clean = []
    seen = set()
    for path in preferred + current:
        if not path or path in seen or any(part in path for part in blocked):
            continue
        clean.append(path)
        seen.add(path)
    return os.pathsep.join(clean)


def _rviz_cmd(filtered_ld, rviz_config):
    snap_vars = [
        'SNAP', 'SNAP_ARCH', 'SNAP_COMMON', 'SNAP_CONTEXT', 'SNAP_COOKIE',
        'SNAP_DATA', 'SNAP_EUID', 'SNAP_INSTANCE_NAME',
        'SNAP_LAUNCHER_ARCH_TRIPLET', 'SNAP_LIBRARY_PATH', 'SNAP_NAME',
        'SNAP_REAL_HOME', 'SNAP_REVISION', 'SNAP_UID', 'SNAP_USER_COMMON',
        'SNAP_USER_DATA', 'SNAP_VERSION', 'GTK_PATH', 'LOCPATH',
    ]
    unset = []
    for var in snap_vars:
        unset.extend(['-u', var])
    return [
        'env', *unset,
        'PATH=%s' % _filtered_path(),
        'LD_LIBRARY_PATH=%s' % filtered_ld,
        'rviz2',
        '-d', rviz_config,
        '--ros-args', '-p', 'use_sim_time:=true',
    ]


def _resolve(share, rel):
    """Prefer the built share/ copy; fall back to the source tree."""
    p = os.path.join(share, rel)
    return p if os.path.exists(p) else os.path.join(SRC_PKG, rel)


def _resolve_generated(rel):
    """Resolve generated assets from the installed package, with source fallback."""
    share = get_package_share_directory('f1tenth_gym_gazebo')
    return _resolve(share, rel)


def _defaults():
    """config/sim.yaml -> {arg: 'string'} for DeclareLaunchArgument defaults."""
    vals = dict(_FALLBACK_DEFAULTS)
    try:
        with open(_resolve_generated(os.path.join('config', 'sim.yaml'))) as f:
            loaded = yaml.safe_load(f) or {}
        vals.update({k: v for k, v in loaded.items() if k in _FALLBACK_DEFAULTS})
    except (OSError, yaml.YAMLError):
        pass                                    # fall back rather than fail to launch
    return {k: (str(v).lower() if isinstance(v, bool) else str(v))
            for k, v in vals.items()}


def _gazebo_params_file():
    """Write gzserver's ROS params to a temp file, sourced from config/sim.yaml.

    config/ holds exactly ONE yaml (sim.yaml), but gzserver needs a real
    `--ros-args --params-file` whose top level is node-name -> ros__parameters;
    sim.yaml's flat launch defaults cannot satisfy that shape, so the file is
    generated instead of being a second checked-in yaml. `-p` is NOT an
    alternative here: that is gzserver's own --play flag.

    publish_rate matters: gazebo_ros_init defaults /clock to 10 Hz, which
    throttles every use_sim_time node (plant 100 Hz, MPC 50 Hz) down with it.
    """
    rate = 200.0
    try:
        with open(_resolve_generated(os.path.join('config', 'sim.yaml'))) as f:
            rate = float((yaml.safe_load(f) or {}).get('gazebo_publish_rate',
                                                       rate))
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        pass                                    # keep the 200 Hz default
    fd, path = tempfile.mkstemp(prefix='f1tenth_gazebo_params_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump({'gazebo': {'ros__parameters': {'publish_rate': rate}}}, f)
    return path


def _all_maps():
    """Base maps + every name in the generated spawns file, so procedurally
    generated maps (rand_*) are launchable without editing this list."""
    names = list(_BASE_MAPS)
    try:
        path = _resolve_generated(
            os.path.join('maps', 'generated_safe_maps_spawns.yaml'))
        with open(path) as f:
            for line in f:
                n = line.split(':', 1)[0].strip()
                if n and '{' in line and n not in names:
                    names.append(n)
    except OSError:
        pass
    return names


MAPS = _all_maps()


def _read_spawn(path, name):
    """Minimal parse of 'name: {x: .., y: .., yaw: ..}' lines (no yaml dep)."""
    with open(path) as f:
        for line in f:
            if line.split(':', 1)[0].strip() == name and '{' in line:
                body = line.split('{', 1)[1].split('}', 1)[0]
                d = {}
                for kv in body.split(','):
                    k, v = kv.split(':')
                    d[k.strip()] = float(v)
                return d
    raise RuntimeError('spawn for %r not found in %s' % (name, path))


def _assert_matching_assets(name, world_file, map_yaml):
    """Fail early if Gazebo and RViz would show different generated maps."""
    with open(world_file) as f:
        world_text = f.read()
    if "<model name='%s'>" % name not in world_text:
        raise RuntimeError(
            'selected world does not contain the requested model %r: %s'
            % (name, world_file))
    for other in MAPS:
        if other != name and "<model name='%s'>" % other in world_text:
            raise RuntimeError(
                'selected world %s also contains generated model %r; refusing '
                'to launch a mixed map' % (world_file, other))

    image_name = None
    with open(map_yaml) as f:
        for line in f:
            if line.strip().startswith('image:'):
                image_name = line.split(':', 1)[1].strip()
                break
    expected = name + '.png'
    if image_name != expected:
        raise RuntimeError(
            'selected map YAML points to %r, expected %r: %s'
            % (image_name, expected, map_yaml))


def _assert_clean_runtime():
    """Avoid mixing worlds/maps from a previous sim_launch still running."""
    checks = [
        ('gzserver', 'Gazebo server'),
        ('gzclient', 'Gazebo client'),
        ('rviz2', 'RViz'),
        ('robot_state_publisher', 'robot state publisher'),
        ('map_publisher', 'map publisher'),
        ('dynamics_plant', 'dynamics plant'),
        ('dynamic_obstacles', 'dynamic obstacles'),
        ('lpv_mpc_node', 'MPC controller'),
        ('ackermann_to_twist', 'Ackermann bridge'),
        ('pose_reset', 'pose reset'),
    ]
    live = []
    for pattern, label in checks:
        proc = subprocess.run(['pgrep', '-fa', pattern],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,
                              text=True,
                              check=False)
        lines = []
        for line in proc.stdout.splitlines():
            # Ignore this launch process if the command text happens to contain
            # one of the patterns in the Python source.
            if 'sim_launch.py' in line or 'pgrep -fa' in line:
                continue
            lines.append(line)
        if lines:
            live.append('%s already running:\n%s' % (label, '\n'.join(lines)))
    if live:
        raise RuntimeError(
            'Refusing to launch while old sim/RViz nodes are still running. '
            'Stop the previous launch with Ctrl-C first.\n'
            + '\n\n'.join(live))


def _filtered_ld():
    cur = os.environ.get('LD_LIBRARY_PATH', '')
    return os.pathsep.join(p for p in cur.split(os.pathsep)
                           if '/snap/core20' not in p)


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory('f1tenth_gym_gazebo')
    name = LaunchConfiguration('map_name').perform(context)
    if name not in MAPS:
        raise RuntimeError('map_name must be one of %s, got %r' % (MAPS, name))
    if LaunchConfiguration('require_clean_runtime').perform(context).lower() == 'true':
        _assert_clean_runtime()

    world_file = _resolve_generated(os.path.join('worlds', name + '.world'))
    spawns = _resolve_generated(os.path.join(
        'maps', 'generated_safe_maps_spawns.yaml'))
    xacro_file = _resolve(share, os.path.join('urdf', 'ego_racecar_gazebo.xacro'))
    gazebo_params = _gazebo_params_file()
    map_yaml = _resolve_generated(os.path.join('maps', name + '.yaml'))
    rviz_config = _resolve(share, os.path.join('rviz', 'f1tenth_gazebo.rviz'))
    for p in (world_file, spawns, xacro_file, map_yaml, rviz_config):
        if not os.path.exists(p):
            raise RuntimeError('missing asset: %s (generate maps / colcon build?)'
                               % p)
    _assert_matching_assets(name, world_file, map_yaml)

    sp = _read_spawn(spawns, name)

    def ov(argname, default):     # user override wins; '' -> spawn default
        v = LaunchConfiguration(argname).perform(context)
        return v if v != '' else str(default)
    x, y, yaw = ov('x', sp['x']), ov('y', sp['y']), ov('yaw', sp['yaw'])
    z = LaunchConfiguration('z').perform(context)
    x_float, y_float, yaw_float = float(x), float(y), float(yaw)

    # Sensor flags are xacro args, so the URDF is rebuilt per launch -> ablating
    # a sensor removes it from Gazebo, not just from the topic list.
    urdf_content = subprocess.check_output([
        'xacro', xacro_file,
        'enable_lidar2d:=' + LaunchConfiguration('enable_lidar2d').perform(context),
        'enable_lidar3d:=' + LaunchConfiguration('enable_lidar3d').perform(context),
        'enable_camera_d435:='
        + LaunchConfiguration('enable_camera_d435').perform(context),
        'enable_imu:=' + LaunchConfiguration('enable_imu').perform(context),
        'camera_mount_z:='
        + LaunchConfiguration('camera_mount_z').perform(context),
        'camera_pitch:=' + LaunchConfiguration('camera_pitch').perform(context),
        'camera_image_width:=' + LaunchConfiguration('camera_image_width').perform(context),
        'camera_image_height:=' + LaunchConfiguration('camera_image_height').perform(context),
        'camera_rate:=' + LaunchConfiguration('camera_rate').perform(context),
    ]).decode('utf-8')
    filtered_ld = _filtered_ld()

    return [
        LogInfo(msg=['sim_launch map_name=', name]),
        LogInfo(msg=['Gazebo world: ', world_file]),
        LogInfo(msg=['RViz /map YAML: ', map_yaml]),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_file,
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so',
                 '--ros-args', '--params-file', gazebo_params],
            output='screen'),
        ExecuteProcess(cmd=['gzclient'], output='screen',
                       condition=IfCondition(LaunchConfiguration('gui'))),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='ego_robot_state_publisher', output='screen',
             parameters=[{'robot_description': urdf_content, 'use_sim_time': True}],
             remappings=[('/robot_description', '/ego_robot_description')],
             condition=IfCondition(LaunchConfiguration('spawn_car'))),
        Node(package='gazebo_ros', executable='spawn_entity.py',
             name='spawn_f1tenth', output='screen',
             arguments=['-entity', 'f1tenth', '-topic', 'ego_robot_description',
                        '-x', x, '-y', y, '-z', z, '-Y', yaw],
             condition=IfCondition(LaunchConfiguration('spawn_car'))),
        Node(package='f1tenth_gym_gazebo', executable='dynamics_plant',
             name='dynamics_plant', output='screen',
             parameters=[{'use_sim_time': True, 'entity_name': 'f1tenth',
                          'x': x_float, 'y': y_float, 'yaw': yaw_float,
                          'spawn_z': float(z),
                          'vibration_pitch_deg': float(LaunchConfiguration('vibration_pitch_deg').perform(context)),
                          'vibration_roll_deg': float(LaunchConfiguration('vibration_roll_deg').perform(context)),
                          'vibration_freq_hz': float(LaunchConfiguration('vibration_freq_hz').perform(context)),
                          'vibration_heave_mm': float(LaunchConfiguration('vibration_heave_mm').perform(context))}],
             condition=IfCondition(LaunchConfiguration('spawn_car'))),
        Node(package='f1tenth_gym_gazebo', executable='map_publisher',
             name='map_publisher', output='screen',
             parameters=[{'use_sim_time': True, 'map_yaml': map_yaml,
                          'topic': '/map', 'frame_id': 'map'}]),
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=_rviz_cmd(filtered_ld, rviz_config),
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('rviz'))),
            ]),
    ]


def generate_launch_description():
    share = get_package_share_directory('f1tenth_gym_gazebo')
    d = _defaults()
    set_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=os.path.join(SRC_PKG, 'models') + os.pathsep
        + _resolve(share, 'models') + os.pathsep
        + os.environ.get('GAZEBO_MODEL_PATH', ''))
    fix_snap_ld = SetEnvironmentVariable(name='LD_LIBRARY_PATH',
                                         value=_filtered_ld())
    fix_python_path = SetEnvironmentVariable(name='PATH',
                                             value=_filtered_path())
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value=d['map_name'],
                              description='one of: ' + ', '.join(MAPS)),
        DeclareLaunchArgument('rviz', default_value=d['rviz'],
                              description='launch RViz2'),
        DeclareLaunchArgument('gui', default_value=d['gui'],
                              description='launch gzclient GUI'),
        DeclareLaunchArgument('spawn_car', default_value=d['spawn_car'],
                              description='spawn the F1TENTH robot'),
        DeclareLaunchArgument(
            'require_clean_runtime', default_value=d['require_clean_runtime'],
            description='refuse to launch if old Gazebo/RViz/map nodes are running'),
        DeclareLaunchArgument('enable_lidar2d', default_value=d['enable_lidar2d'],
                              description='Enable 2D LiDAR'),
        DeclareLaunchArgument('enable_lidar3d', default_value=d['enable_lidar3d'],
                              description='Enable 3D LiDAR'),
        DeclareLaunchArgument('enable_camera_d435',
                              default_value=d['enable_camera_d435'],
                              description='Enable D435 camera'),
        DeclareLaunchArgument('enable_imu', default_value=d['enable_imu'],
                              description='Enable IMU'),
        DeclareLaunchArgument('camera_mount_z',
                              default_value=d['camera_mount_z'],
                              description='Camera height above ground [m]. '
                                          'lane_assist cam_height must match.'),
        DeclareLaunchArgument('camera_pitch', default_value=d['camera_pitch'],
                              description='Camera down-pitch [rad]. '
                                          'lane_assist cam_pitch must match.'),
        DeclareLaunchArgument('camera_image_width', default_value=d['camera_image_width'],
                              description='RGB image width [px] (hfov fixed at 69.4 deg).'),
        DeclareLaunchArgument('camera_image_height', default_value=d['camera_image_height'],
                              description='RGB image height [px]; 360 = the real D435i 16:9 colour stream.'),
        DeclareLaunchArgument('camera_rate', default_value=d['camera_rate'],
                              description='RGB frame rate [Hz]; the D435i streams 30.'),
        DeclareLaunchArgument('x', default_value='',
                              description='override spawn X (default: map spawn)'),
        DeclareLaunchArgument('y', default_value=''),
        DeclareLaunchArgument('z', default_value='0.0'),
        DeclareLaunchArgument('vibration_pitch_deg', default_value='0.0',
                              description='Chassis pitch vibration amplitude [deg] '
                                          '(camera + lidar wobble; 0 = off).'),
        DeclareLaunchArgument('vibration_roll_deg', default_value='0.0',
                              description='Chassis roll vibration amplitude [deg].'),
        DeclareLaunchArgument('vibration_freq_hz', default_value='2.0',
                              description='Vibration base frequency [Hz] (pose written at 100 Hz: up to ~30 Hz).'),
        DeclareLaunchArgument('vibration_heave_mm', default_value='0.0',
                              description='Chassis heave (vertical bounce) amplitude [mm].'),
        DeclareLaunchArgument('yaw', default_value=''),
        set_model_path,
        fix_snap_ld,
        fix_python_path,
        OpaqueFunction(function=launch_setup),
    ])
