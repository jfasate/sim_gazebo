# Keyboard Teleoperation for F1Tenth Gazebo Simulation

Integration of keyboard teleoperation control for manual vehicle operation in the simulation.

## Usage

### Launch the Teleoperation Node

```bash
# In a new terminal, ensure ROS2 is sourced
source install/setup.bash

# Run the teleop node
ros2 run f1tenth_gym_gazebo teleop_keyboard
```

## Keyboard Controls

```
Moving around:
   u    i    o
   j    k    l
   m    ,    .

anything else : stop

Speed Controls:
  q/z : increase/decrease both accelerator and steering speeds by 10%
  w/x : increase/decrease only accelerator speed by 10%
  e/c : increase/decrease only steering speed by 10%

CTRL-C to quit
```

## Published Topics

- **`/accelerator_cmd`** (std_msgs/msg/Float32): Accelerator pedal command (-1.0 to 1.0)
- **`/steering_cmd`** (std_msgs/msg/Float32): Steering command (-1.0 to 1.0)

## Parameters

The teleop node supports the following ROS2 parameters (can be set via command line or config file):

- **`accelerator_pedal_cmd_topic`** (string, default: "accelerator_cmd"): Topic name for accelerator commands
- **`steering_cmd_topic`** (string, default: "steering_cmd"): Topic name for steering commands
- **`init_pedal_cmd_mag`** (float, default: 0.5): Initial accelerator command magnitude
- **`init_steer_cmd_mag`** (float, default: 1.0): Initial steering command magnitude
- **`update_frequency`** (float, default: 10.0): Keyboard polling frequency in Hz

### Example with Parameters

```bash
ros2 run f1tenth_gym_gazebo teleop_keyboard \
  --ros-args \
  -p init_pedal_cmd_mag:=0.3 \
  -p init_steer_cmd_mag:=0.5 \
  -p accelerator_pedal_cmd_topic:=/drive_command \
  -p steering_cmd_topic:=/steer_command
```

## Typical Workflow

1. **Terminal 1**: Start the Gazebo simulation
   ```bash
   source install/setup.bash
   ros2 launch f1tenth_gym_gazebo sim_launch.py rviz:=false
   ```

2. **Terminal 2**: Start the teleoperation controller
   ```bash
   source install/setup.bash
   ros2 run f1tenth_gym_gazebo teleop_keyboard
   ```

3. **Use keyboard commands** to control the vehicle

4. **Press CTRL-C** in the teleop terminal to stop and gracefully shut down

## Notes

- The script works best with a US keyboard layout
- The keyboard input is non-blocking with a 0.1s timeout (10 Hz polling)
- Commands are published at the configured update frequency
- Sending 'CTRL-C' properly shuts down the node and publishes zero commands before exiting
