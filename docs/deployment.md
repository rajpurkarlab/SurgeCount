# Deployment

**Datasets and checkpoints will be released upon publication.** The release
configuration targets a single-camera pi0.5 policy and uses a generic local
checkpoint placeholder. Supply a local checkpoint containing `config.json`,
`model.safetensors`, and the saved policy pre/postprocessor JSON and
normalization state files. Provision the tokenizer cache in the target
environment separately. Real deployment sets the Hugging Face libraries
to offline mode and does not download model artifacts.

```bash
surgecount-deploy --config configs/deploy.json --hardware configs/robot.example.json --dry-run
surgecount-deploy --config configs/deploy.json --hardware /path/to/robot.json --checkpoint /path/to/checkpoint --check
surgecount-deploy --config configs/deploy.json --hardware /path/to/robot.json --checkpoint /path/to/checkpoint
```

Both check modes leave hardware disconnected. `--check` validates local
calibration/home files and checkpoint metadata, including pi05, thirteen
action dimensions, and camera keys. It does not instantiate the large
policy or establish that inference will meet its timing target.

The default mode is manual skill selection. For automated selection,
install the `planner` extra, export `GEMINI_API_KEY` in the environment,
and add `--mode gemini`. No `.env` file is automatically loaded. The
planner sends the three camera views and skill history to the configured
Gemini service; this is enabled only in Gemini mode. The supplied model
identifier records the workstation launcher setting and can be changed in
the JSON when that API model is unavailable.

Each policy invocation uses a twelve-dimensional robot state, three image
slots, and the exact skill instruction. Two policy image slots are blacked
out according to the training camera map. The planner inspects all three
unmasked views. `image_keys: openpi` reproduces the workstation launcher's
renaming into `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`;
use `native` only with checkpoints that declare native `bi_so_follower`
image keys. The checker rejects a mismatch.
New checkpoints trained with this release's native-camera training recipe
use `image_keys: native`; the supplied historical deployment preset uses
`openpi` for the preserved workstation checkpoint.

The policy outputs twelve robot targets and one stop scalar. Only the
twelve robot targets reach the motors. The controller keeps the full
policy tensor for stop detection and holds the inactive arm at its
subtask-start pose. Every task transition clears the policy action queue,
processors, interpolation, stop state, and arm snapshots.

The bundled configuration preserves the later workstation settings:

| Setting | Value |
| --- | --- |
| Observation/action rate | 10 Hz nominal |
| Interpolation | 3× (nominal 30 Hz motor updates) |
| Denoising steps | 20 |
| Default action chunk | 40 |
| Skill 1 and 6 chunks | 30 → 25 → 20 → 15 → 10 → 5 → 30, reset per skill |
| Stop rule | First five-prediction window above 0.5 within a predicted chunk |
| Timeouts | Skills 1/6: 60 s; skill 5: 30 s |

Lookahead stops at the beginning of the predicted high-stop window.
Queued normalized actions are postprocessed into the same stop-target
space as the current action before comparison. This differs from the
older manual launcher, which required ten consecutive executed stop
predictions: to reproduce that rule, use `stop_mode: consecutive`,
`stop_window: 10`, disable `dynamic_chunk.enabled`, and choose an
interpolation multiplier of 1. This changes the stop rule; it does not
establish that an earlier experimental run used the rest of this config.

On completion the controller holds position. Gemini then selects a new
skill or waits; manual mode waits for the operator. Slot occupancy updates
only when release finishes immediately after a slot-placement skill.
Timeout handling holds for inspection and can use the externally supplied
home pose for right-arm recovery. Enter saves an episode, Escape ends the
session, `e` pauses, and the digit/slot keys select skills. Physical motor
updates may be slower than nominal rates because of inference, camera,
or planner latency; no real-time guarantee is implied.
Keyboard input received during a planner request invalidates that response.
Pending suggestions are cleared by pause or exit, and the planner waits
while the operator selects a slot after pressing `2`.

The vendored runtime is an integration of the preserved workstation
sources with the recorded local LeRobot baseline. Offline tests exercise
configuration validation and compatibility fixes. Robot motion, GPU
throughput, and live Gemini behavior require validation on the actual
setup before using a run as experimental evidence.
