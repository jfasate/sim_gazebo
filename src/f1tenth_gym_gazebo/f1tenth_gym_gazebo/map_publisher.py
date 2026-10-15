#!/usr/bin/env python3
"""Minimal, self-contained occupancy-grid map publisher.

Loads a map_server-format map (.yaml + image) and publishes it once as a
latched nav_msgs/OccupancyGrid on /map, so RViz shows the walls and the
"2D Pose Estimate" click lands in a known place relative to them. Avoids the
nav2 map_server lifecycle stack (not installed here) — same output, no deps
beyond PIL/numpy.

Occupancy encoding (map_server convention, negate=0):
    occ_prob = (255 - pixel) / 255
    occ_prob > occupied_thresh -> 100 (occupied)
    occ_prob < free_thresh     ->   0 (free)
    otherwise                  ->  -1 (unknown)        mode: trinary (default)
                                   1..99 (scaled)      mode: scale
'scale' keeps in-between pixels as graded occupancy, so a map can draw
features gray in RViz without the plant's collision guard treating them as
unknown (= occupied): only cells >= its threshold (50) block the car.

Parameters:
    map_yaml  : path to the .yaml
    topic     : output topic (default /map)
    frame_id  : map frame (default map)
"""

import math
import os

import numpy as np
from PIL import Image
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class MapPublisher(Node):

    def __init__(self):
        super().__init__('map_publisher')

        self.declare_parameter('map_yaml', '')
        self.declare_parameter('topic', '/map')
        self.declare_parameter('frame_id', 'map')

        yaml_path = self.get_parameter('map_yaml').value
        topic = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value

        if not yaml_path or not os.path.isfile(yaml_path):
            raise SystemExit(f'map_yaml not found: "{yaml_path}"')

        grid = self._load(yaml_path)

        # Latched (transient-local) QoS so RViz gets the map on connect.
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, topic, qos)
        self.grid = grid
        self.pub.publish(grid)

        # Re-publish occasionally so late/again-subscribing tools still get it.
        self.timer = self.create_timer(2.0, self._republish)
        self.get_logger().info(
            f'Published map on {topic} ({grid.info.width}x{grid.info.height} '
            f'@ {grid.info.resolution} m/px, frame="{self.frame_id}")')

    def _republish(self):
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.grid)

    def _load(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        img_name = meta['image']
        img_path = img_name if os.path.isabs(img_name) else \
            os.path.join(os.path.dirname(yaml_path), img_name)
        res = float(meta['resolution'])
        origin = meta.get('origin', [0.0, 0.0, 0.0])
        negate = int(meta.get('negate', 0))
        occ_th = float(meta.get('occupied_thresh', 0.65))
        free_th = float(meta.get('free_thresh', 0.196))
        mode = str(meta.get('mode', 'trinary'))

        img = Image.open(img_path).convert('L')
        px = np.asarray(img, dtype=np.float32)      # (H, W), row 0 = top
        # OccupancyGrid data starts at the lower-left corner -> flip vertically.
        px = np.flipud(px)

        if negate:
            occ = px / 255.0
        else:
            occ = (255.0 - px) / 255.0

        data = np.full(occ.shape, -1, dtype=np.int8)
        if mode == 'scale':
            data[:] = (99.0 * (occ - free_th) / (occ_th - free_th)).astype(np.int8)
        data[occ > occ_th] = 100
        data[occ < free_th] = 0

        H, W = data.shape
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = res
        grid.info.width = W
        grid.info.height = H
        p = Pose()
        p.position.x = float(origin[0])
        p.position.y = float(origin[1])
        qx, qy, qz, qw = yaw_to_quat(float(origin[2]) if len(origin) > 2 else 0.0)
        p.orientation.x, p.orientation.y = qx, qy
        p.orientation.z, p.orientation.w = qz, qw
        grid.info.origin = p
        grid.data = data.reshape(-1).tolist()
        return grid


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
