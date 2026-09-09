# Hardware and data collection

The setup uses two SO-101 follower arms, two leader arms for teleoperation,
and three RGB cameras (left front, overhead, right front). Record camera
frames at 640 × 480 and demonstrations at 10 Hz. The example uses MJPG to
reduce USB bandwidth; the right camera runs at 30 Hz while recording consumes
observations at 10 Hz.

Copy `configs/robot.example.json` to a private machine configuration outside
the repository. Replace every placeholder with the ports and camera paths
for that machine. On Linux, stable `/dev/serial/by-id` and `/dev/v4l/by-id`
paths avoid numeric index changes after reconnecting. Other platforms can
use serial device paths and integer OpenCV camera indices.

Calibration and home poses are machine-specific and are not distributed.
The follower calibration directory must contain `bimanual_left.json` and
`bimanual_right.json`; the leader directory must contain
`bimanual_leader_left.json` and `bimanual_leader_right.json`. If IDs change,
filenames must change accordingly. Use independently verified calibration
for each physical arm. The release controller connects with
`calibrate=False` and never initiates interactive calibration. Confirm that
motor calibration matches the saved files before deployment.

`use_degrees` is explicit for each arm. The example retains the source
configuration's degree convention for joint positions; grippers use their
configured normalized range. This setting must match the demonstration
data and trained policy. `max_relative_target` is available for a
machine-specific motion limit; the example does not invent a limit.

The optional home file is text containing one `joint_name=value` per line
for all twelve joint/gripper targets. Joint names are `left_` and `right_`
plus `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`,
`wrist_flex.pos`, `wrist_roll.pos`, and `gripper.pos`. No joint angles are
provided. Removing `home_pose_file` disables between-episode homing; a
home pose is required when enabling timeout recovery or Gemini's
`ra_home` skill. The wrapper does not perform startup homing.

From the repository root, inspect a launch before connecting:

```bash
surgecount-record --config configs/record.json --hardware configs/robot.example.json --dry-run
surgecount-record --config configs/record.json --hardware /path/to/robot.json --check
surgecount-record --config configs/record.json --hardware /path/to/robot.json
```

`--dry-run` validates schema and prints the argument vector. `--check` also
reads the local calibration/home files. Neither mode imports hardware
drivers, opens devices, or records data. Actual recording requires the
robot extra, FFmpeg, and access to the configured devices.

During teleoperation, digits select the per-frame language instruction;
press `2` then `a`, `b`, or `c` for slots 1, 2, or 3. `e` ends a subtask;
`0` marks the preceding attempt as dropped. Dropped segments and their
failed predecessor are relabeled idle when saving. Enter saves the episode,
Backspace rerecords, Escape ends the session, and `h` requests the configured
home pose and a new episode. Raw recordings have twelve action dimensions;
stop targets are constructed during dataset preparation.
The collection wrapper preregisters all thirteen canonical task labels so
dataset task IDs remain stable regardless of the first skill recorded.
Keyboard digits and dataset task IDs are different; see `tasks.py`.

Local recordings are never uploaded by the wrappers. Choose a fresh
`dataset.root`; use `resume: true` only to append to an existing compatible
dataset. Run recordings interactively in a terminal with keyboard access.
