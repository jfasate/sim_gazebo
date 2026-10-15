# F1Tenth Sim — Sensor-Suite Upgrade Roadmap

Goal: evolve the current 2D-LiDAR + single-camera rig into a **generic AV sensor
superset** (3D LiDAR + surround cameras + GNSS + IMU) that can be ablated
down to any subset, and use it to **generate labeled perception-training data**.

Decisions locked (2026-08-19):
- **Reference suite:** generic superset (ablate any subset).
- **Primary goal:** perception training data.
- **Migration appetite:** stay on **Gazebo Classic 11 now**, migrate to Harmonic
  *later*, gated on a real need for dense per-pixel masks (Phase 6 only).

---

## 0. Current state (ground truth)

| Fact | Value |
|---|---|
| Simulator | **Gazebo Classic 11.10.2** (`gzserver`/`gzclient`, `libgazebo_ros_*.so`) |
| ROS | Humble |
| Launch | [launch/sim_launch.py](../launch/sim_launch.py) |
| Robot desc | [urdf/ego_racecar_gazebo.xacro](../urdf/ego_racecar_gazebo.xacro) |
| Motion | **Co-sim** — [dynamics_plant.py](../f1tenth_gym_gazebo/dynamics_plant.py) integrates a double-track+Pacejka model, writes pose via `/set_entity_state`; gravity off on all links; **no drive plugin** |
| World state | Each world loads `libgazebo_ros_state.so` → serves `/get_entity_state`, `/set_entity_state`, publishes `/gazebo/model_states` + `/gazebo/link_states` |
| Obstacles | Represented as **links inside one world model** (e.g. `offset_obstacles` → `Wall_0..3`, `Obstacle_4..5`), box geometry in the `.world` SDF |

Current sensors (all in the xacro):

| Sensor | Spec | Topic | Rate |
|---|---|---|---|
| 2D LiDAR (Hokuyo) | 1080 pts, 270°, 30 m | `/scan` | 20 Hz |
| RGB (D435 color) | 640×480, ~69° HFOV | `/camera/color/image_raw` | 15 Hz |
| Depth (D435) | 640×480 float32 | `/camera/depth/image_raw` | 15 Hz |
| IMU | accel+gyro, gaussian noise | `/imu` | 100 Hz |

**Stale config to fix:** `config/bridge_config.yaml` and `package.xml` describe a
"Gazebo Harmonic + ros_gz_bridge" setup that **does not exist**. Dead config; Phase 0
relabels it.

---

## 1. Invariants (must hold in every phase)

1. **Gazebo is a sensor renderer only.** Never add a drive plugin, never re-enable
   gravity, never touch the `/set_entity_state` write-back. New sensors are extra
   render outputs on the existing body — architecturally free, no new physics.
2. **Sensors are opt-in.** Every sensor block is a `xacro:macro` gated by a
   `xacro:arg` boolean so any subset can be launched (the "ablate" requirement).
3. **Measure RTF after every sensor add.** GPU render budget — not CPU physics — is
   the ceiling here. Keep the 200 Hz `/clock` from `config/gazebo_params.yaml`
   (the 10 Hz default aliases high-rate sensors).
4. **Run under system Python 3.10**, not conda 3.13 (the launch already filters
   conda/snap from PATH — new nodes inherit that; don't bypass it).

## 2. The insight that defers migration

The co-sim holds **perfect ground-truth world state**. `/gazebo/link_states` gives
live poses of every wall/obstacle/agent; the `.world` SDF gives their box dims.
Therefore these labels are generable **on Classic, today, with no segmentation
camera**:

- 3D bounding boxes (all obstacles/agents)
- 2D boxes (project 3D corners through each camera's `camera_info` K matrix)
- BEV occupancy-grid truth (map + dynamic links)
- Depth truth (existing depth cam)
- Agent/ego trajectories (prediction/planning labels)

The **only** label Classic cannot make is dense per-pixel semantic/instance masks —
that, and nothing else, is what justifies the Harmonic migration in Phase 6. So
nothing built in Phases 0–5 is wasted.

---

## 3. Target sensor superset & frame tree

```
ego_racecar/base_link
├── laser_model            (existing 2D lidar, /scan)
│   └── camera_link        (existing D435 front: color + depth)
├── lidar_3d_link          [P1] roof-center 3D LiDAR   → /lidar/points
├── cam_front_link         [P3] surround ring          → /camera/front/*
├── cam_front_left_link    [P3]                         → /camera/front_left/*
├── cam_front_right_link   [P3]                         → /camera/front_right/*
├── cam_rear_left_link     [P3]                         → /camera/rear_left/*
├── cam_rear_right_link    [P3]                         → /camera/rear_right/*
├── cam_rear_link          [P3]                         → /camera/rear/*
├── gnss_link              [P4]                         → /gnss/fix
└── imu_link               (existing, /imu)
```

Extrinsics are the URDF joint origins — dump them + intrinsics to JSON in Phase 5
so downstream training has calibration.

---

## Phase 0 — De-risk & make sensors ablatable  *(~0.5 day)*

**Goal:** stop trusting dead config; turn the monolithic xacro into per-sensor
macros with enable flags.

**Files:** `urdf/ego_racecar_gazebo.xacro`, `config/bridge_config.yaml`,
`package.xml`, `launch/sim_launch.py`.

**Steps:**
1. Rename `config/bridge_config.yaml` → `config/bridge_config.harmonic.yaml` and add
   a top comment: *"UNUSED on Classic. Goes live only in Phase 6 (Harmonic)."* Fix
   the `package.xml` `<description>` to say Gazebo Classic 11.
2. Wrap each existing sensor (`lidar2d`, `camera_d435`, `imu`) in a
   `<xacro:macro>`; add `<xacro:arg name="enable_<sensor>" default="true"/>` and
   `<xacro:if value="$(arg enable_<sensor>)">` guards.
3. Pass args through in `sim_launch.py`: build the xacro command with
   `Command(['xacro ', xacro_file, ' enable_lidar3d:=', LaunchConfiguration(...)])`
   instead of the current `subprocess.check_output(['xacro', xacro_file])`, and
   declare matching `DeclareLaunchArgument`s.

**Acceptance:** `ros2 launch … enable_camera_d435:=false` starts with no
`/camera/color` topic; default launch is byte-for-byte behavior-identical to today.

**Risk:** the launch currently shells `xacro` to a string; switching to `Command`
substitution needs `ParameterValue(..., value_type=str)`. Low risk, well-trodden.

---

## Phase 1 — 3D LiDAR  *(~1 day — highest value, Classic-native, zero new deps)*

**Goal:** velodyne-style point cloud from the stock ray sensor.

**Files:** `urdf/ego_racecar_gazebo.xacro` (+ macro from P0).

**Implementation** — add a `lidar_3d_link` on the roof and a GPU ray sensor with a
vertical scan block; the stock `libgazebo_ros_ray_sensor.so` emits `PointCloud2`:

```xml
<gazebo reference="${car_name}/lidar_3d_link">
  <sensor name="lidar_3d" type="gpu_ray">   <!-- gpu_ray = GPU-accelerated -->
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal><samples>900</samples><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
        <vertical>  <samples>16</samples> <min_angle>-0.2618</min_angle> <max_angle>0.2618</max_angle></vertical>  <!-- ±15°, VLP-16 -->
      </scan>
      <range><min>0.3</min><max>100.0</max><resolution>0.01</resolution></range>
      <noise><type>gaussian</type><mean>0.0</mean><stddev>0.008</stddev></noise>
    </ray>
    <plugin name="lidar_3d" filename="libgazebo_ros_ray_sensor.so">
      <output_type>sensor_msgs/PointCloud2</output_type>
      <frame_name>${car_name}/lidar_3d_link</frame_name>
      <ros><remapping>~/out:=/lidar/points</remapping></ros>
    </plugin>
  </sensor>
</gazebo>
```

Channel count is an `xacro:arg` (16→32→64) so you can trade fidelity for RTF.

**Topics/Frames:** `/lidar/points` (PointCloud2), frame `lidar_3d_link`.

**Acceptance:** `ros2 topic hz /lidar/points` ≈ 10 Hz; cloud renders in RViz aligned
with walls; RTF stays > 0.9 at 16 ch.

**Ceiling / defer:** `# ponytail: stock ray sensor — uniform ring, no per-beam
intensity/dropout. Swap to velodyne_simulator plugin only if ring realism matters.`

---

## Phase 2 — Ground-truth label pipeline  *(~2–3 days — the actual goal)*

**Goal:** one node turning co-sim world state into perception labels. **This is what
"perception training data" needs, and it needs no migration.**

**New file:** `f1tenth_gym_gazebo/gt_labeler.py` (+ entry point in `setup.py`).
**Dep:** `vision_msgs` (add to `package.xml`).

**Inputs:**
- `/gazebo/link_states` (`gazebo_msgs/LinkStates`) — live pose of every obstacle link.
- Box dims parsed once from the active `.world` SDF (`<box><size>` per collision
  link). Pass world path as a param; cache a `{link_name: (dx,dy,dz)}` map.
- `camera_info` per camera (intrinsics K) + TF (extrinsics) for 2D projection.
- Ego pose from `/ego_racecar/odom`.

**Outputs:**
| Product | Message | Topic |
|---|---|---|
| 3D boxes | `vision_msgs/Detection3DArray` | `/labels/boxes_3d` |
| 2D boxes (per cam) | `vision_msgs/Detection2DArray` | `/labels/<cam>/boxes_2d` |
| BEV occupancy truth | `nav_msgs/OccupancyGrid` | `/labels/bev_occupancy` |
| Agent tracks | `vision_msgs/Detection3DArray` (w/ ids) | `/labels/tracks` |

**2D projection math** (per camera, per 3D box):
1. 8 box corners in world frame (from link pose + half-dims).
2. Transform world→camera_optical via `tf2` lookup.
3. Keep corners with `z_cam > 0`; project `u = fx·x/z + cx`, `v = fy·y/z + cy`.
4. 2D bbox = min/max of projected `(u,v)`, clipped to image; drop if fully outside.

**Acceptance:** a self-check `gt_labeler_test.py` — place one known box in front of
one camera, assert the projected 2D bbox center is within a few px of the analytic
value; assert 3D box count == obstacle-link count. (`assert`-based `__main__`, no
framework.)

**Ceiling:** `# ponytail: axis-aligned world boxes from SDF <box> dims only; meshes
approximated by AABB. No per-pixel masks (that's Phase 6).`

---

## Phase 3 — Surround camera ring  *(~1–2 days — value, but the GPU cost center)*

**Goal:** 6-camera ring (extend to 8 for Tesla-style by adding two calls).

**Files:** `urdf/ego_racecar_gazebo.xacro`.

**Implementation:** one macro, called per camera:

```xml
<xacro:macro name="surround_cam" params="name x y z yaw fov:=1.51 rate:=10 w:=640 h:=480">
  <link name="${car_name}/cam_${name}_link"/>
  <joint name="cam_${name}_joint" type="fixed">
    <parent link="${car_name}/base_link"/><child link="${car_name}/cam_${name}_link"/>
    <origin xyz="${x} ${y} ${z}" rpy="0 0 ${yaw}"/>
  </joint>
  <gazebo reference="${car_name}/cam_${name}_link">
    <sensor name="cam_${name}" type="camera">
      <update_rate>${rate}</update_rate>
      <camera><horizontal_fov>${fov}</horizontal_fov>
        <image><width>${w}</width><height>${h}</height><format>R8G8B8</format></image>
        <clip><near>0.1</near><far>50.0</far></clip>
      </camera>
      <plugin name="cam_${name}" filename="libgazebo_ros_camera.so">
        <camera_name>camera/${name}</camera_name>
        <frame_name>${car_name}/cam_${name}_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</xacro:macro>

<xacro:surround_cam name="front"       x="0.30" y="0.0"   z="0.20" yaw="0"      w="1280" h="720" rate="15"/>
<xacro:surround_cam name="front_left"  x="0.20" y="0.10"  z="0.20" yaw="1.05"/>
<xacro:surround_cam name="front_right" x="0.20" y="-0.10" z="0.20" yaw="-1.05"/>
<xacro:surround_cam name="rear_left"   x="0.0"  y="0.10"  z="0.20" yaw="2.09"/>
<xacro:surround_cam name="rear_right"  x="0.0"  y="-0.10" z="0.20" yaw="-2.09"/>
<xacro:surround_cam name="rear"        x="-0.05" y="0.0"  z="0.20" yaw="3.14159"/>
```

**Perf discipline:** front cam high-res @ 15 Hz; the other five 640×480 @ 10 Hz.
Rate/res are macro params so you dial the GPU budget. **Measure RTF before/after** —
this is where it drops.

**Acceptance:** 6 `/camera/*/image_raw` topics at target rates; RTF > 0.8 with all
six + 3D LiDAR; each `camera_info` present (feeds Phase 2 projection).

---

## Phase 4 — GNSS  *(~0.5 day)*

**GNSS** — no reliable stock GPS plugin on Classic+ROS2. A ~15-line node adds
gaussian noise to the ground-truth ego pose and emits `sensor_msgs/NavSatFix` on
`/gnss/fix` (co-sim pose makes this trivial and *more* correct than a plugin).
- New file `f1tenth_gym_gazebo/gnss_sim.py`; local-tangent-plane → lat/lon around a
  configurable datum.

**Acceptance:** `/gnss/fix` tracks ego with configured noise σ.

*(Radar dropped from scope 2026-08-19 — no native Classic sensor and the geometric
fake added little over the 3D LiDAR. Re-add later if a fusion stack needs range-rate.)*

---

## Phase 5 — Recorder, calibration, sim-to-real knobs  *(~2 days)*

**Goal:** make the data actually usable for training.

1. **Calibration dump:** small node/script writes intrinsics (from `camera_info`) +
   extrinsics (from TF) to `calibration.json` once at startup — KITTI/nuScenes-ish.
2. **Synchronized recorder:** start with plain `ros2 bag record` of the sensor +
   `/labels/*` topics (works day 1, zero code). Add a structured
   frame-folder exporter (`message_filters.ApproximateTimeSynchronizer` →
   per-frame `{clouds, images, boxes}`) **only when** a training loader needs it.
   `# ponytail: rosbag first; structured exporter only when a loader demands it.`
3. **Sim-to-real noise/latency:** **reuse the existing
   [sensor_delay.py](../f1tenth_gym_gazebo/sensor_delay.py)** pattern per sensor —
   the delay/noise knob already exists; don't rebuild it.

**Acceptance:** one `ros2 bag` replays with clouds+images+labels time-aligned;
`calibration.json` round-trips through the Phase 2 projection.

---

## Phase 6 — *(DEFERRED)* Harmonic migration — dense masks only

**Trigger:** an actual need for per-pixel semantic/instance masks. Not sooner.

**Scope when triggered:**
- Port sensor blocks from `libgazebo_ros_*.so` to gz-sim `<plugin>` systems.
- Replace `/set_entity_state` write-back with the gz Entity/pose service in
  `dynamics_plant.py` (the one real co-sim change).
- Bring `config/bridge_config.harmonic.yaml` live via `ros_gz_bridge`.
- Add `segmentation_camera` + `boundingbox_camera` (the only new capability).
- Keep the Classic launch working in parallel until parity is proven.

Everything from Phases 1–5 (3D boxes, 2D boxes, occupancy, GNSS, recorder)
carries over unchanged — this phase *only* adds dense masks.

---

## Perf budget (the real constraint)

| Config | Approx GPU load | Expected RTF |
|---|---|---|
| Today (2D lidar + 1 cam) | low | ~1.0 |
| + 3D LiDAR 16 ch @ 10 Hz | +144k pts/s | > 0.9 |
| + 6-cam ring (1×720p15 + 5×480p10) | **dominant** | 0.7–0.85 |
| + GNSS | negligible | ~same |

Mitigations already baked in: per-sensor enable flags (ablate), rate/res/channel
xacro args, GPU ray sensor, 200 Hz `/clock` retained. **Rule: re-check RTF after
each phase; if < 0.8, drop a camera rate before adding the next sensor.**

## Deferred / ceiling ledger

| Item | Ceiling | Upgrade path |
|---|---|---|
| 3D LiDAR | uniform ring, no intensity/dropout | velodyne_simulator plugin |
| GT boxes | AABB from SDF `<box>`; meshes approximated | mesh-accurate hull |
| Dense masks | **not available on Classic** | Phase 6 (Harmonic) |
| Recorder | rosbag first | structured exporter when loader needs it |

## Open decisions (revisit before Phase 3/6)

- Camera count: 6 (Waymo/Zoox-ish) vs 8 (Tesla). Ring macro makes it a 2-line change.
- LiDAR channels: 16 (cheap) vs 32/64 (heavier) — pick per RTF headroom after P3.
- Recording format: rosbag2 vs nuScenes/KITTI folders — defer to real training loader.
