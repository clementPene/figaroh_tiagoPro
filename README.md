# Figaroh — Tiago Pro geometric calibration

Identifies per-joint angle offsets for the right arm chain
`base_link → gripper_right_tool_holder` using external EE pose measurements
(Qualisys mocap). Results are written to `calibration_offset.urdf.xacro`,
the same format used by the PAL calibration system.

**Goal**: reduce the FK/mocap error at the source so that `mocap_mpc_corrector.py`
applies near-zero correction → no more MPC instability.

---

## Setup

### 1. Clone robot description packages (meshes)

```bash
cd /home/cpene/Documents/figaroh_tiagoPro
vcs import < tiago_pro_description.repos
```

### 2. Create UV environment

```bash
uv venv
source .venv/bin/activate
uv pip install pin viser picos pyyaml matplotlib trimesh
uv pip install -e figaroh/
```

---

## Workflow

### Step 1 — Generate the URDF (container)

With the robot launched, run from inside the container:

```bash
ros2 run xacro xacro \
  $(ros2 pkg prefix tiago_pro_description)/share/tiago_pro_description/robots/tiago_pro.urdf.xacro \
  end_effector_right:=pal-atc \
  end_effector_left:=pal-pro-gripper \
  wrist_model_right:=spherical-wrist \
  > /home/gepetto/ros2_ws/src/figaroh_tiagoPro/tiago_pro.urdf
```

Then post-process on the host to make mesh paths portable:

```bash
python3 generate_urdf.py   # tiago_pro.urdf → tiago_pro_local.urdf
```

Alternatively, to get the calibrated URDF (with current offsets applied),
use `save_urdf.py` from the demo 07 package instead of xacro:

```bash
# In container, robot launched
python3 /home/gepetto/ros2_ws/src/agimus-demos/agimus_demo_07_fixed_tiago_pro_deburring/scripts/save_urdf.py \
  --output /home/gepetto/ros2_ws/src/figaroh_tiagoPro/tiago_pro.urdf
```

### Step 2 — Visualize the robot (host)

```bash
source .venv/bin/activate
python3 view_robot.py
```

Opens Viser at http://localhost:8080.

### Step 3 — Generate optimal calibration configurations (host)

```bash
source .venv/bin/activate
python3 generate_optimal_configs.py
```

- Generates a pool of random collision-free configurations
- Selects the D-optimal subset (maximizes information for parameter identification)
- Visualizes each config in Viser (Enter = next, q = quit)
- Saves to `data/optimal_configs.yaml`

Options:
```bash
python3 generate_optimal_configs.py --pool-size 500 --no-viser
```

### Step 4 — Collect calibration data (container, robot running)

```bash
# In container, robot launched + Qualisys running
python3 /home/gepetto/ros2_ws/src/agimus-demos/agimus_demo_07_fixed_tiago_pro_deburring/scripts/collect_calibration_data.py \
  --configs /home/gepetto/ros2_ws/src/figaroh_tiagoPro/data/optimal_configs.yaml
```

- Sends `FollowJointTrajectory` goals to `arm_right_controller` + `torso_controller`
- Waits for the robot to settle at each configuration
- Records `(q, T_mocap_EE_in_base)` from `/joint_states` + `/mocap_ee_pose`
- Saves to `data/calibration_samples.csv`

### Step 5 — Run calibration (host)

```bash
source .venv/bin/activate
python3 run_calibration.py \
  --urdf tiago_pro_local.urdf \
  --data data/calibration_samples.csv
```

Outputs:
- Identified joint offsets + marker position (RMSE/MAE in mm)
- Writes results to `calibration_offset.urdf.xacro`

### Step 6 — Apply on robot

Copy the generated xacro to the robot:

```bash
scp calibration_offset.urdf.xacro \
  user@tiago-pro:/etc/ros/urdf/calibration/calibration_offset.urdf.xacro
```

Then restart the robot description:

```bash
pal module_manager restart robot_state_publisher
```

---

## Identified parameters

| Parameter | Description |
|---|---|
| `offsetRZ_arm_right_1..7_joint` | Per-joint angle offset (rad) |
| `offsetPZ_torso_lift_joint` | Torso linear offset (m) |
| `pEEx_1, pEEy_1, pEEz_1` | Mocap marker position rel. to `gripper_right_tool_holder` (m) |

---

## Files

| File | Where | Description |
|---|---|---|
| `generate_urdf.py` | host | Post-process container URDF → portable |
| `view_robot.py` | host | Visualize robot in Viser |
| `generate_optimal_configs.py` | host | D-optimal config selection |
| `collect_calibration_data.py` | container (demo 07) | Move robot + record data |
| `run_calibration.py` | host | Figaroh LM calibration |
| `save_urdf.py` | container | Dump `/robot_description` to file |
| `tiago_pro_calibration_config.yaml` | host | Figaroh config |
| `tiago_pro.srdf` | host | Collision pairs (from demo 07) |
| `tiago_pro_description.repos` | host | VCS repos for robot meshes |
