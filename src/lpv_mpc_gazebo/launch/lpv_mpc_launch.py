import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _resolve_log_dir():
    """Return the source-tree log/ folder (created if missing).

    The node runs from the install space, but we want the per-run CSV logs in
    the source package (src/lpv_mpc_gazebo/log) so they are easy to inspect and are
    not wiped by `colcon build`. Prefer the LPV_MPC_LOG_DIR env override, then
    the conventional ~/sim_gazebo/src/lpv_mpc_gazebo/log path.
    """
    log_dir = os.environ.get(
        'LPV_MPC_LOG_DIR',
        os.path.expanduser('~/sim_gazebo/src/lpv_mpc_gazebo/log'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lpv_mpc_gazebo'),
        'config',
        'lpv_mpc_params.yaml')

    log_dir = _resolve_log_dir()

    lpv_mpc_node = Node(
        package='lpv_mpc_gazebo',
        executable='lpv_mpc_node',
        name='lpv_mpc_node',
        output='screen',
        # config first, then the resolved log_dir override so every run lands
        # in the source-tree log/ folder (enable_csv_log stays YAML-controlled).
        parameters=[config, {'log_dir': log_dir, 'config_file': config}],
    )

    nodes = [lpv_mpc_node]

    # In ref_source='topic' mode, also start the node that feeds the controller's
    # /planning/ref_path with a rolling local horizon sliced from the CSV — the
    # non-ML stand-in for the generator. ref_generator 'external' starts NOTHING
    # so an outside node (e.g. neural_planner/neural_generator) owns the topic
    # without two publishers fighting over it. csv mode is unchanged.
    with open(config) as f:
        params = yaml.safe_load(f)['lpv_mpc_node']['ros__parameters']
    external = str(params.get('ref_generator', 'builtin')).strip().lower() == 'external'
    if str(params.get('ref_source', 'csv')).strip().lower() == 'topic' and not external:
        nodes.append(Node(
            package='lpv_mpc_gazebo',
            executable='window_republisher',
            name='window_republisher',
            output='screen',
            parameters=[{
                'reference_csv': params.get('reference_csv', 'test_worldv5'),
                'ref_path_topic': params.get('ref_path_topic', '/planning/ref_path'),
            }],
        ))

    return LaunchDescription(nodes)
