# SurgeCount

**A Protocol-Guided Composition Harness for Bimanual Surgical Sponge Counting**

Research code prepared for ML4H 2026.

SurgeCount uses two SO-101 arms to move individual surgical sponges from a
source tray into ordered counting slots. A single pi0.5 policy learns eleven
language-conditioned skills and their stopping signal. An operator or a
stateful vision-language planner chooses the next skill at each boundary.

This repository contains the data collection, data preparation, training,
manual deployment, and Gemini composition code, plus an anonymous project
website. It includes the patched LeRobot source needed by the runtime.

**Datasets and checkpoints will be released upon publication.** This anonymous
review release contains no project dataset or checkpoint links. Local examples
allow the code to be inspected and used with independently collected data.

## Installation

Use **Python 3.12**, FFmpeg (including `ffprobe`), and `uv`. Run commands from
this repository root. The dependency lock pins `transformers==5.3.0` and uses
the local LeRobot snapshot rather than an incompatible public package.

```bash
# Data preparation, offline checks, and tests
uv sync --frozen --no-editable --python 3.12 --extra data --extra dev

# Training and physical robot deployment; planner is optional
uv sync --frozen --no-editable --python 3.12 --extra data --extra dev --extra train --extra robot --extra planner
```

The full environment installs PyTorch and substantial model dependencies.
Training uses an NVIDIA GPU; the recovered experiment used an H200. The
deployment example targets CUDA. MPS is configurable, but was not validated
as part of this release. Provide platform-appropriate FFmpeg and device
drivers before using video or hardware commands. Activate the environment
with `source .venv/bin/activate` to use the bare command names in the guides.
These release installs use ordinary packages. After changing source, repeat
the sync command with `--reinstall-package surgecount` (and
`--reinstall-package lerobot` for vendor changes).

## Quick start

All four commands support `--help`. These dry runs print validated
configuration without opening robot ports, starting training, or calling a
cloud model:

```bash
uv run --no-sync surgecount-record --config configs/record.json --hardware configs/robot.example.json --dry-run
uv run --no-sync surgecount-train --config configs/train.json --dry-run
uv run --no-sync surgecount-deploy --config configs/deploy.json --hardware configs/robot.example.json --dry-run
```

1. **Collect demonstrations.** Copy `configs/robot.example.json` to an ignored
   `configs/robot.local.json`, fill in the actual ports/cameras and existing
   verified calibration/home files, then run the recording command with
   `--check`. Remove `--check` to collect leader-follower teleoperation.
   Digit keys label skills; `2` then `a/b/c` selects a slot. See
   [hardware and recording](docs/hardware.md).
2. **Prepare a new dataset copy.** Inspect your LeRobot v3 data with
   `surgecount-data inspect`. Add the thirteenth action target with
   `surgecount-data add-stop`; source data is never edited. See
   [data format, cleaning, and stop labels](docs/data.md).
3. **Train.** Point `configs/train.json` at a locally prepared dataset. Inspect
   the dry-run report, then launch with `surgecount-train`. See
   [training](docs/training.md) for episode filtering, normalization, and
   pretrained initialization.
4. **Deploy.** Provision a local checkpoint and tokenizer cache, complete the
   hardware configuration, and run `surgecount-deploy --check` before the
   real command. Manual skill selection is the default. For the automated
   harness, add `--mode gemini` and export `GEMINI_API_KEY`. See
   [deployment](docs/deployment.md) for stop rules, timeouts, camera names,
   arm holding, and operator controls.

Commands that start recording/deployment operate real motors. Dry runs and
`--check` remain offline; calibration is never regenerated on connect.

## Final training recipe

| Item | Recovered final setup |
| --- | --- |
| Dataset | Curated bimanual gauze demonstrations; release upon publication |
| Initialization | `lerobot/pi05_base`, full finetuning |
| Episode subset | After 250; release default is `episode_index >= 251` |
| Sampling | Exclude idle task 0 and drop task 8 as training anchors |
| Observations | Twelve joint/gripper states; three recorded RGB views |
| Policy images | One camera per skill; other two images blacked out |
| Targets | Twelve robot actions plus stop; final ten segment frames have stop=1 |
| Batch / precision | 192 / bfloat16, gradient checkpointing |
| Checkpoint interval | Every 1,000 steps |

The release config runs 10,000 steps. That stopping point and inherited
optimizer settings are documented release defaults, not a recovered
byte-for-byte launch configuration. Idle/drop filtering applies to sampled
anchor frames; native temporal action chunks may extend across task
boundaries. New release-trained checkpoints retain native camera names;
the historical deployment preset expects OpenPI names. The deployment
checker requires the selected names to match the checkpoint.

Artifact availability and local checkpoint requirements are described in
[models](docs/models.md). The code release does not contain raw demonstrations
or model weights. The curated dataset and eligible training subset are
distinct from the original recording collection; see the [lineage](docs/data.md).

## Repository map

```text
src/surgecount/        Collection/deployment launchers, dataset tools, training hooks
configs/              Portable recording, training, deployment, and hardware examples
vendor/lerobot/        Patched LeRobot source, upstream license, change record
tests/                Offline data, filtering, runtime, and website regressions
docs/                 Setup, data lineage, models, provenance, and validation
website/              Self-contained anonymous ML4H project page and demo media
tools/                Release and anonymous-website checks
```

## Checks

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests tools
uv run --no-sync python tools/check_release.py
uv run --no-sync python tools/check_website.py --media
```

CI runs offline tests and artifact checks without credentials or hardware.
See [validation](docs/validation.md) for what was tested and what still
requires the original robot, dataset, or checkpoints.

## Website and licensing

The code and website are prepared for anonymous review; see the
[anonymous release notes](docs/anonymity.md). Serve only `website/` as the
web artifact. The website contains no author,
institution, personal account, manuscript, or repository links and makes no
third-party resource requests. See [website notes](docs/website.md).

First-party code and the LeRobot snapshot use Apache-2.0. The website layout
and third-party assets have separate terms in [THIRD_PARTY.md](THIRD_PARTY.md).
Use `CITATION.cff` for the current anonymous software citation; no acceptance
or publication identifier is claimed.
