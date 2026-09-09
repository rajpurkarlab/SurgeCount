# Vendored LeRobot source

This directory contains the LeRobot Python source and package metadata needed
for the SurgeCount collection, training, and deployment entry points.
The package declares version 0.5.2 and Python >= 3.12. It is a preserved
working-source snapshot with local research changes. Private Git history
and source commit identifiers are withheld for anonymous review; this
vendored tree is the source used by the release.

Upstream: https://github.com/huggingface/lerobot

The Apache-2.0 license is preserved in `LICENSE`, and source copyright
headers and upstream package authors remain intact. Upstream source files
may include additional third-party notices, which are retained.

The later workstation bundle supplied updated `lerobot_record.py`,
`control_utils.py`, `gemini_planner.py`, and `evdev_keyboard.py`. They were
placed in the corresponding `lerobot.scripts`, `lerobot.common`, and
`lerobot.utils` namespaces. Generic planner result types and the skill
vocabulary were extracted to `common/planner_types.py`; the obsolete
Claude API implementation is not included. Planner instructions embedded
in `gemini_planner.py` are the runtime's task-specific planning logic.
Private development conversations, API keys, machine configuration,
calibration, home angles, traces, recordings, and model weights are not
part of this directory.

Release integration changes:

- The record configuration accepts canonical task labels, preregisters their
  IDs, validates resume mappings, and records ended subtasks as idle.
- Connection defaults to `calibrate=False`; wrappers provide explicit
  machine configuration, home-file paths, and local output directories.
- pi0.5 checkpoint loading propagates failures instead of returning an
  uninitialized policy. Explicit Hub/cache arguments are respected.
- `ActionInterpolator` supports the workstation runtime's observed-pose
  priming and per-boundary interpolation multipliers.
- Every changed subtask flushes the action queue, processors, interpolation,
  stop counters, chunk schedule, and inactive-arm snapshots.
- When the configured chunk length changes, an empty pi0.5 queue is resized
  before replenishment, preserving the beginning of the predicted chunk.
- Planner results carry an operator-input revision and are installed under
  the keyboard event lock. Inputs received during an API call invalidate
  the response; pause/exit clear pending suggestions, and a pending slot
  choice prevents automatic planning.
- Stop extraction retains the thirteenth action component independently
  of the twelve motor commands. Lookahead decodes queued normalized action
  targets before applying the stop threshold. At a stop boundary, the held
  command uses observed joints instead of the pending unsent target.
- The consecutive-stop mode supports the earlier manual controller's
  stop rule. Lookahead is covered by the tested `common/stop_signal.py`.
- Robot command failures propagate instead of being recorded as successful
  sends. Check modes live in the outer SurgeCount wrappers and do not call
  this hardware runtime.
- The package readme field now references the included `README.md` directly.

Checks performed during integration: all Python files parsed with Python
3.12; the updated recorder imported with the available PyTorch/LeRobot
dependencies; generated recording arguments parsed into the actual
dataclass configuration without creating hardware. The release tests run
without hardware and include configuration, interpolation compatibility,
stop-window, and failed-weight-loading regression cases. No physical
robot, checkpoint weights, live planner service, or training run was used
for these checks.
