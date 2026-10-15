# lane_assist — ViT lane assist

A Vision Transformer turns **one camera image** into the **reference path** the car
follows: lane keeping, a lane change around an obstacle, or a stop in front of a full
road block. Pure pursuit or the LPV-MPC (`lpv_mpc_gazebo`, topic mode) follows the path;
a 3D-lidar safety brake limits the speed.
The classical lane detector was removed on 2026-10-08 (its baseline copy for the
report lives in `~/sim_gazebo/eval/vit_report/classical_lane_detector.py`).

Scope: daytime (noon) only, simulation (ROS 2 Humble, Gazebo 11, F1Tenth scale).
The detailed model walkthrough is in [docs/VIT_LANE_ASSIST.md](docs/VIT_LANE_ASSIST.md);
the results report is `~/sim_gazebo/eval/vit_report/RESULTS.html`.

## Layout

```
lane_assist/                 runtime (ROS nodes + shared libraries)
  vit_lane.py                ViT-Ti/16 model (PyTorch port of google-research/vision_transformer)
  vit_lane_node.py           camera -> ViT -> /planning/ref_path (+ stop from the 17th output)
  lane_follow_node.py        pure pursuit on the ego-frame path (+ lidar speed cap)
  lidar_guard.py             /lidar/points -> free distance (path corridor + 1.8 m straight hard stop)
  road_collect.py            data-collection node (episodes keep / avoid / late / stop / wrong)
  eval_tools.py              map ground truth: the expert plan (labels), free distance, metrics
  camera_geometry.py         pixel <-> ground, horizon row, ego <-> world
launch/vit_follow_launch.py  vit_lane_node + follower (arg controller: pure_pursuit | lpv_mpc) (+ lidar_guard)
config/lane_assist_params.yaml   ALL node parameters (road_collect, vit_lane_node, lidar_guard, and the
                                 lpv_mpc_node lane-mode overrides on top of lpv_mpc_params.yaml)
tools/maps/make_road_maps.py     map generator (writes worlds/models/maps into f1tenth_gym_gazebo)
tools/maps/truth/<map>.npz       map ground truth (lanes, obstacles, side lanes)
tools/data/collect_all.py        collect many maps (Gazebo windows, parallel, resumable)
tools/data/make_dataset.py       labels from poses -> dataset/<map>_labels.csv
tools/train/train_vit.py         training -> models/vit_lane_<N>maps[_all].pt
tools/analysis/analyze_run.py    score a log/vit_run_* folder
test/                            pytest + launch_testing (see below)
docs/                            model walkthrough
models/ dataset/ log/            data, not in git
```

## Data flow

```
camera 640x480 -> vit_lane_node (crop below horizon, 384x128, ViT, 17 outputs, EMA)
   -> /planning/ref_path (N,6) [s, x, y, psi, kappa, vx], EGO frame
   -> lane_follow_node (pure pursuit 50 Hz) or lpv_mpc_node (MPC 50 Hz) -> /drive
/lidar/points -> lidar_guard -> /lane_assist/lidar_free -> follower speed cap
```

- Output 0..15: lateral offset Y [m, + left] of the path at XS = 0, 1/3, ..., 5 m.
  It is where the path IS (lane centre or lane change), not where the car is.
- Output 16: free distance x FREE_SCALE (0.25); 8 m = clear. Speed is capped at
  sqrt(2 * brake_decel * (free - stop_margin)).
- Dynamic speed (vx column): cruise `target_speed` on a clear straight, sqrt(a_lat_max/|kappa|)
  per point, braked back from slow points ahead; followers use vx at the car (row 0).
  Training data was collected at 0.8-2.0 m/s: raise the cruise in steps and re-test.

## The one source of truth for "correct"

`eval_tools.Truth.plan_label(x, y, yaw)` — collector, labels and tests all use it:
lane centre; or a 5.5 m smoothstep lane change finished 1.5 m before an in-path
obstacle into a free, directly adjacent same-direction lane (left first; never bike /
bus / parking; ended lanes ignored); `late_label` for a car still in its lane close to
the obstacle; `free_distance` for a full road block (no free lane: stay and stop).
Labels need only the pose, so `make_dataset.py` can relabel all data at any time.

## Commands

```bash
cd ~/sim_gazebo && source /opt/ros/humble/setup.bash && source install/setup.bash

# drive it (two terminals)
ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:=road_test
ros2 launch lane_assist vit_follow_launch.py              # lidar_guard:=false = model only
ros2 launch lane_assist vit_follow_launch.py controller:=lpv_mpc

# offline pipeline (from src/lane_assist)
/usr/bin/python3 tools/maps/make_road_maps.py 4 105        # maps road_105..108 (then colcon build f1tenth_gym_gazebo)
/usr/bin/python3 tools/data/collect_all.py road_105 --n 2000 --parallel 4
/usr/bin/python3 tools/data/make_dataset.py $(seq -f "road_%02g" 0 99) $(seq -f "road_%g" 100 108) --test
/usr/bin/python3 tools/train/train_vit.py $(seq -f "road_%02g" 0 99) $(seq -f "road_%g" 100 108) --all --epochs 25 --no-flip

# tests (from src/lane_assist)
/usr/bin/python3 -m pytest test/test_late_label.py test/test_stop_and_wrong_side.py -q
# closed loop (from ~/sim_gazebo); fails on ANY contact or > 0.5 s on the wrong side
LANE_TEST_ONLY=test_6 LANE_TEST_SEED=0 LANE_TEST_EPISODES=10 LANE_TEST_EPISODE_S=60 \
  launch_test src/lane_assist/test/test_closed_loop.py
#   LANE_TEST_MAP=road_test_stop   held-out stop map;   LANE_TEST_GUARD=false   model alone
#   LANE_TEST_CONTROLLER=lpv_mpc   MPC follower (default pure_pursuit)
```

## Maps

Families are NESTED levels (`make_road_maps.family(seed)`, `at(seed, fam)`):
general < obstacle < hard < cone < bare (unpainted) < crash (debris, ghost lines, 1 full
block/direction) < block (3 full blocks in one direction, passables in the other) < decoy
(crash + 60% unpainted + cone/barrel rows on the shoulders of curves, shadow bands; decoys
kept >= 0.2 m outside every lane) < coneblock (block, but every full block = cone rows in
every lane). Seeds 0..108 by range (0..47 general, 48..71 obstacle, 72..83 hard, 84..92 cone,
93..98 bare, 99..104 crash, 105..108 block); seeds 109..252 from `EXTRA_FAMILY` (block
109..120, crash 121..130, bare 131..138, cone 139..146, hard 147..152, decoy 153..202,
coneblock 203..252). New maps: add seeds to `EXTRA_FAMILY`. Test maps, never collected:
`road_test` (13 scripted sections, `--test`) and `road_test_stop` (`--test-stop`).
`collect_all.py` only globs `road_[0-9]*`. Never change obstacles on a map whose frames
are collected; new map families take new seeds (old maps must reproduce exactly).

## Rules that bite

- `cam_height` / `cam_pitch` / `cam_x` (vit_lane_node section) must equal the sim mount;
  make_dataset.py and train_vit.py read them from there. The crop starts at
  `camera_geometry.horizon_row(K, pitch) + 2`.
- Train with `--no-flip`: a mirrored image is left-hand traffic (taught wrong-side driving).
- Stop frames are rare: keep `--stop-weight` (20); without it the 17th output collapsed.
- Collection is GUI only (Gazebo + RViz windows), never headless. Parallel workers get
  their own ROS_DOMAIN_ID and Gazebo port.
- All parameters live in `config/lane_assist_params.yaml` (plain `declare_parameter`).
- LPV-MPC lane mode needs no map/localization: the path is ego-frame; odom gives the body
  speeds and the motion since the last path. Its topic-mode code is in
  `lpv_mpc_gazebo/lpv_mpc_node.py` (ref timeout, lidar cap, odom motion); CSV racing mode is untouched.
- Always kill every sim process after a run.
