# Training SurgeCount skills

The release trains individual language-conditioned gauze-handling skills with twelve robot commands and a stop signal. The operator/Gemini deployment harness selects and sequences those skills into a complete counting procedure.

The end-to-end pi0.5 baseline uses a pi0-based policy with a trained subtask predictor, following the original pi0.5 paper (arXiv:2504.16054). At each action chunk boundary, the overhead image and umbrella instruction produce subtask text that conditions twelve-dimensional action generation, with no stop signal. Its predictor-training branch and exact objective remain pending source recovery. The training commands below cover the thirteen-dimensional SurgeCount skill policy.

## Final recipe and reconstruction limits

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
| Finetuning | All model parameters trainable; no PEFT, frozen vision encoder, or expert-only mode |
| Batch | 192 per process; historical setup used one GPU |
| Precision | bfloat16 with gradient checkpointing |
| Dataset decoding | PyAV |
| Seed | 1000 |
| Release run length / save interval | 10,000 updates / every 1,000 updates |

**Datasets and checkpoints will be released upon publication.** Reported
checkpoint steps are 4,000, 6,000, 9,000, and 10,000. Their account names,
repository identifiers, and download links are withheld for anonymous review.

The original complete final launch configuration was not recovered. The 10,000-update default reaches the highest reported checkpoint step. The inclusive episode boundary, optimizer/scheduler settings, action chunk size, and worker settings in `configs/train.json` are explicit reconstruction choices. The optimizer defaults come from the vendored `pi05` preset. LeRobot scales scheduler timing for shorter runs; the effective pipeline configuration is saved with checkpoints.

The example leaves `base_model` unset. Supply a compatible local initialization checkpoint with `--base-model`; its configuration and weights must load through the `pi05` interface. Record dataset and initialization revisions in your private run configuration for future reproduction. For Hub-backed artifacts, `dataset_revision` and `base_revision` accept immutable commit hashes.

Ordinary initialization values name local directories. An intentional Hub
initialization uses `hf://` followed by a namespace and repository name.
Missing initialization or an invalid local checkpoint fails before model
imports or downloads.

## Prepare and inspect

Install the training environment using the main README. Training is intended for a sufficiently large CUDA GPU; the historical batch 192 full finetuning setup used an H200. This release has not rerun that expensive training or verified it on other hardware.

First prepare the action13 dataset using [data.md](data.md). The default `local/gauze_training` identifier is a local placeholder, not a Hub repository. It requires an explicit `dataset_root` for a real run; missing local data fails before Hub or model imports. Use a local tree so its content can be audited before model loading:

```bash
surgecount-train --config configs/train.json \
  --dataset-root /path/to/prepared_dataset \
  --base-model /path/to/initialization_checkpoint --dry-run
```

The dry run prints the complete release recipe and, when given a local root, computed local task/frame counts. It does not contact the Hub, load model weights, allocate a GPU model, start training, or open robot devices. The local dry-run audit counts eligible anchors; full per-row stop-label and normalization validation runs before actual model creation.

Start a real run after reviewing the data:

```bash
surgecount-train --config configs/train.json \
  --dataset-root /path/to/prepared_dataset \
  --base-model /path/to/initialization_checkpoint \
  --output-dir outputs/train/skill_singlecam_run1
```

An existing output directory is refused. This entry point starts a new finetuning run and saves locally. Provision the initialization checkpoint and its tokenizer before starting a run with local artifacts.

## What the implementation checks

Task strings must exactly match the canonical task table. State must have 12 dimensions and actions 13. Every action must be finite, stop labels must match contiguous segment ends, and normalization statistics must match the full feature widths. Exactly the three expected cameras must exist. An empty filtered subset fails immediately.

The filtering wrapper selects **positions in the already loaded LeRobot table**, including its episode subset. It never confuses a global frame index with a local subset position. LeRobot's original table remains available for temporal action and video queries. Masking happens on each fetched sample, before collation and model preprocessing; it allocates black inactive images without mutating source samples. A top-level `masked_collate` is also available for custom loaders and supports spawned workers.

“Idle/drop excluded” follows the recovered **anchor-sampling** implementation. A selected anchor's future action chunk can still cross into another task or idle frames within the same episode. This release preserves that behavior and does not claim every future target frame was removed. Segment clipping or temporal loss masking would be a changed training condition.

The `pi05` checkpoint interface retains a padded action projection width (normally 32), so a 13-dimensional target does not require resizing compatible pretrained projection tensors. Dataset features specify the 13 valid output dimensions; LeRobot pads internally and trains on those valid dimensions. The entire initialization checkpoint is loaded strictly through the vendored fail-closed loader. Weight errors stop the run.

Pre/postprocessors are built fresh using this dataset's 13-dimensional action statistics and 12-dimensional state statistics. This avoids accidentally loading a base checkpoint's mismatched normalizers or relative-action transform. Robot actions remain absolute, and the language input is the current subtask instruction.

## Run artifacts and deployment

`surgecount_run.json` records the recipe, assumptions, resolved dataset directory, eligible episode IDs, and actual anchor counts by task. LeRobot saves model configuration, model weights, pre/postprocessor configurations and state, and optimizer/scheduler state in its checkpoint directories.

New release-trained checkpoints use the native dataset camera feature names. Set deployment `image_keys` to `native` when using them. Historical checkpoints can use the older OpenPI camera names; the deployment checkpoint checker validates that choice. Match the training camera mask, prompt text, joint order, action normalization, and stop interpretation during deployment.

The tests exercise synthetic v3 parquet conversion across shards, metadata/statistics preservation, first-occurrence task canonicalization, sparse episode subset indexing, camera masks, worker serialization, and scoped processor integration. They do not establish real-robot performance, learned policy quality, or reproduction of manuscript results.
