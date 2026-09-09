# Training SurgeCount skills

SurgeCount trains language-conditioned gauze-handling skills with twelve robot commands and a stop signal. The operator/Gemini harness selects and sequences those skills into a complete counting procedure.

The commands below train the thirteen-dimensional skill policy through LeRobot's `pi05` checkpoint interface. The separate end-to-end pi0.5 baseline combines a pi0-based policy with a trained subtask predictor; its architecture and source status are described in [models.md](models.md).

## Training recipe

| Condition | Setting |
| --- | --- |
| Dataset | Curated bimanual gauze demonstrations; release upon publication |
| Initial weights | Explicit compatible checkpoint supplied with `--base-model` |
| Checkpoint interface | LeRobot `pi05` |
| Targets | 12 robot commands plus stop at dimension 12 |
| Stop targets | Final ten frames of each contiguous skill segment |
| Episode subset | Default `episode_index >= 251`, inclusive |
| Excluded sampled anchors | Task IDs 0 and 8 |
| Images | One camera per subtask; other two images blacked out |
| Finetuning | All model parameters trainable; PEFT and freeze flags disabled |
| Batch | 192 per process; historical setup used one GPU |
| Precision | bfloat16 with gradient checkpointing |
| Dataset decoding | PyAV |
| Seed | 1000 |
| Release run length / save interval | 10,000 updates / every 1,000 updates |

**Datasets and checkpoints will be released upon publication.** Reported checkpoint steps are 4,000, 6,000, 9,000, and 10,000. Artifact identifiers and download links will accompany that release.

`configs/train.json` defines the release defaults. Its inclusive episode boundary, optimizer/scheduler settings, action chunk size, and worker settings are reconstruction choices; exact reproduction also requires the original final launch configuration. The 10,000-update default reaches the highest reported checkpoint step, and the optimizer defaults come from the vendored `pi05` preset. LeRobot scales scheduler timing for shorter runs and saves the effective pipeline configuration with checkpoints.

Supply a compatible local initialization checkpoint with `--base-model`; the example leaves this field unset. Its configuration and weights must load through the `pi05` interface. Record dataset and initialization revisions in your private run configuration. For Hub artifacts, `dataset_revision` and `base_revision` accept immutable commit hashes.

Ordinary initialization values name local directories. An explicit Hub initialization uses `hf://` followed by a namespace and repository name. Initialization validation checks local files before model imports or downloads.

## Prepare and inspect

Install the CUDA training environment using the main README. The historical batch 192 full finetuning run used one H200.

Prepare the action13 dataset using [data.md](data.md). The default `local/gauze_training` identifier requires an explicit `dataset_root` for a real run. The trainer checks the local tree before Hub or model imports. Audit its contents with:

```bash
surgecount-train --config configs/train.json \
  --dataset-root /path/to/prepared_dataset \
  --base-model /path/to/initialization_checkpoint --dry-run
```

The dry run reads configuration and local dataset columns only. It prints the recipe and, when given a local root, task/frame counts and eligible anchor counts. Training validates every stop label and normalization statistic before model construction.

Start a real run after reviewing the data:

```bash
surgecount-train --config configs/train.json \
  --dataset-root /path/to/prepared_dataset \
  --base-model /path/to/initialization_checkpoint \
  --output-dir outputs/train/skill_singlecam_run1
```

Each finetuning run requires a new output directory and saves locally. Provision the initialization checkpoint and its tokenizer before starting with local artifacts.

## Sampling and model inputs

Validation enforces the canonical task strings, 12-dimensional state, 13-dimensional finite actions, stop labels at contiguous segment ends, complete normalization statistics, and exactly the three expected cameras. The filtered subset must contain eligible anchors.

The filtering wrapper selects **local row positions in the loaded LeRobot table**, including episode subsets. LeRobot resolves temporal action and video queries against the original table. Each fetched sample receives black images for inactive cameras before collation and model preprocessing, while source samples remain unchanged. The top-level `masked_collate` supports custom loaders and spawned workers.

Idle/drop filtering applies to **sampled anchors**. Future action chunks retain LeRobot's episode-based temporal queries and can cross task or idle boundaries within an episode. Segment clipping or temporal loss masking defines a different training condition.

The `pi05` interface retains its padded action projection width, normally 32. Dataset features select the 13 valid output dimensions; LeRobot pads internally and computes loss on those dimensions. Compatible initialization tensors keep their original shapes. Strict weight loading stops the run on weight errors.

Fresh pre/postprocessors use the dataset's 13-dimensional action statistics and 12-dimensional state statistics. Actions are absolute, and the language input is the current subtask instruction.

## Run artifacts and deployment

`surgecount_run.json` records the recipe, assumptions, resolved dataset directory, eligible episode IDs, and actual anchor counts by task. LeRobot saves model configuration, model weights, pre/postprocessor configurations and state, and optimizer/scheduler state in its checkpoint directories.

New release-trained checkpoints use the native dataset camera feature names. Set deployment `image_keys` to `native` when using them. Historical checkpoints can use the older OpenPI camera names; the deployment checkpoint checker validates that choice. Match the training camera mask, prompt text, joint order, action normalization, and stop interpretation during deployment.

Regression tests cover synthetic v3 parquet conversion across shards, metadata/statistics preservation, first-occurrence task canonicalization, sparse episode subset indexing, camera masks, worker serialization, and scoped processor integration. Their validation scope is software behavior on synthetic data and mocked interfaces.
