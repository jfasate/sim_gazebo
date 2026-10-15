#!/usr/bin/env python3
"""Generate a 2D occupancy-grid map (.png + .yaml) from a Gazebo world.

Parses the box-wall geometry of a model in an SDF world and rasterises the
walls into a ROS map_server occupancy grid — no simulation / SLAM run needed,
so the map is exact and reproducible. Output matches the f1tenth_gym_ros map
format (white = free, black = occupied, 0.05 m/px, standard yaml).

Usage:
    /usr/bin/python3 world_to_map.py \
        --world <world.sdf> --model test_worldv5 \
        --out-dir <maps/> --name test_worldv5 \
        [--resolution 0.05] [--margin 0.5]

The map frame equals the Gazebo world frame, so a robot spawned at world (0,0)
appears at world (0,0) on the map (yaml `origin` is the lower-left corner).
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def _strip_ns(root):
    """Remove XML namespaces so plain tag names work (SDF is namespace-free,
    but be safe)."""
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return root


def _pose(el):
    """Parse a <pose> child -> (x, y, yaw). Missing -> zeros."""
    p = el.find('pose')
    if p is None or not p.text:
        return 0.0, 0.0, 0.0
    v = [float(x) for x in p.text.split()]
    x = v[0] if len(v) > 0 else 0.0
    y = v[1] if len(v) > 1 else 0.0
    yaw = v[5] if len(v) > 5 else 0.0
    return x, y, yaw


def find_model(root, name):
    """Return the model element that actually defines geometry (not a <state>
    snapshot): the one whose links contain <collision>/<box>."""
    candidates = [m for m in root.iter('model') if m.get('name') == name]
    for m in candidates:
        if m.find('.//collision/geometry/box') is not None:
            return m
    return candidates[0] if candidates else None


def extract_walls(model):
    """Return list of (cx, cy, yaw, length, thickness) in world coords."""
    mx, my, myaw = _pose(model)
    c, s = math.cos(myaw), math.sin(myaw)
    walls = []
    for link in model.findall('link'):
        box = link.find('.//collision/geometry/box/size')
        if box is None or not box.text:
            continue
        size = [float(v) for v in box.text.split()]
        length, thickness = size[0], size[1]
        lx, ly, lyaw = _pose(link)
        # transform link pose into world by the model pose
        wx = mx + c * lx - s * ly
        wy = my + s * lx + c * ly
        walls.append((wx, wy, myaw + lyaw, length, thickness))
    return walls


def wall_corners(cx, cy, yaw, length, thickness):
    hl, ht = length / 2.0, thickness / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((-hl, -ht), (hl, -ht), (hl, ht), (-hl, ht)):
        pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return pts


def build_map(walls, resolution, margin):
    # bounds over all wall corners
    xs, ys = [], []
    for w in walls:
        for px, py in wall_corners(*w):
            xs.append(px)
            ys.append(py)
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin

    W = int(math.ceil((xmax - xmin) / resolution))
    H = int(math.ceil((ymax - ymin) / resolution))

    img = Image.new('L', (W, H), 255)  # 255 = free (white)
    draw = ImageDraw.Draw(img)

    def to_px(x, y):
        col = (x - xmin) / resolution
        row = (H - 1) - (y - ymin) / resolution  # flip y (image row 0 = top)
        return col, row

    for w in walls:
        poly = [to_px(px, py) for px, py in wall_corners(*w)]
        draw.polygon(poly, fill=0)  # 0 = occupied (black)

    origin = [xmin, ymin, 0.0]
    return img, origin, (W, H)


def write_yaml(path, image_name, resolution, origin):
    with open(path, 'w') as f:
        f.write(f'image: {image_name}\n')
        f.write(f'resolution: {resolution:.6f}\n')
        f.write(f'origin: [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, help='SDF world file')
    ap.add_argument('--model', default='test_worldv5', help='model name')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default=None, help='output basename')
    ap.add_argument('--resolution', type=float, default=0.05)
    ap.add_argument('--margin', type=float, default=0.5)
    args = ap.parse_args(argv)

    name = args.name or args.model
    os.makedirs(args.out_dir, exist_ok=True)

    root = _strip_ns(ET.parse(args.world).getroot())
    model = find_model(root, args.model)
    if model is None:
        raise SystemExit(f'model "{args.model}" not found in {args.world}')

    walls = extract_walls(model)
    if not walls:
        raise SystemExit('no box walls found in model')

    img, origin, (W, H) = build_map(walls, args.resolution, args.margin)

    png_path = os.path.join(args.out_dir, f'{name}.png')
    yaml_path = os.path.join(args.out_dir, f'{name}.yaml')
    img.save(png_path)
    write_yaml(yaml_path, f'{name}.png', args.resolution, origin)

    print(f'walls: {len(walls)}')
    print(f'map:   {W} x {H} px @ {args.resolution} m/px '
          f'({W*args.resolution:.1f} x {H*args.resolution:.1f} m)')
    print(f'origin: [{origin[0]:.3f}, {origin[1]:.3f}, 0.0]')
    print(f'wrote: {png_path}')
    print(f'wrote: {yaml_path}')


if __name__ == '__main__':
    main()
