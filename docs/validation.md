# Release validation

Validation covers the prepared source and static website. It does not
constitute a new robot experiment or reproduction of the paper's results.

## Checks performed

- A fresh, isolated Python3.12 environment was resolved from `uv.lock` and
  installed with the data/development extras. The lock resolves PyTorch2.10.0
  for the optional full environment and pins transformers5.3.0.
- The collection, training, and deployment console scripts passed dry runs
  from an ordinary package installation. The data console script exposes
  both audit and fresh-copy preparation commands.
- 101 offline tests passed in the available test environment with PyTorch;
  ten additional unittest subtests passed. In the isolated Python3.12 data
  environment, 99 tests passed and the two PyTorch-dependent cases skipped.
- The suite covers real Parquet/Arrow metadata, shard-spanning stop labels,
  source-file hash preservation, canonical task remapping, action13 global
  and per-episode statistics, sparse episode indexing, camera masking,
  dataloader behavior, and spawn-compatible collate functions.
- Runtime checks cover offline configuration, calibration/checkpoint
  preflight, stop lookahead, interpolation, failed checkpoint loading,
  dynamic action-queue resizing, and operator priority over late planner
  responses. They use fixtures and stubs rather than physical devices.
- All vendored Python files parse under Python3.12. The recorder and trainer
  import using the existing robot environment with transformers5.3.0.
  Generated recording/deployment arguments parse through the actual LeRobot
  dataclasses. A training-config integration test uses mocked metadata and
  weight-loading entry points, without downloading or constructing a model.
- First-party lint, release-content scanning, website local-target checks,
  and image/video metadata checks pass. Root and vendored packages build.
- Anonymity regressions cover account links, personal paths, hidden metadata,
  binary identifier scans, and exact ZIP membership and neutral metadata.
  Local training placeholders cannot trigger a remote dataset lookup.
  Video metadata checks accept the neutral zero vendor field emitted by
  older FFmpeg versions and continue to reject identifying vendor values.
- The page was inspected in a browser at wide and narrow viewport sizes.
  All eight videos decode successfully. The supplied clip video bitstreams
  are unchanged; container metadata and timecode tracks were stripped.

## Limits and unresolved provenance

No training job, GPU inference, physical robot connection, real pretrained
weight load, or paid planner API call was made during release preparation.
The full train/robot dependency lock was resolved, while the fresh
installation test used the lightweight data/development extras. Native
device libraries and GPU wheels need validation on the deployment machine.

Datasets and checkpoints will be released upon publication. The anonymous
release includes neither artifact links nor copies of the training data or weights.
Exact final episode-boundary conventions, immutable Hub revisions, and the
deleted historical launch configuration are not recovered. The example
config states its defaults and enforces dimensional/task/camera consistency.
Its successful dry run is not evidence that an external artifact has been
downloaded or verified.

Video visual review sampled six frames from each clip, plus the
posters. The architecture figure is retained following visual review,
including its training inset containing a partial operator profile.
Source and media metadata scans complement that review. No manuscript,
private chat history, calibration, credentials, or raw recordings are in
the distributable source artifact. The complete source archive and website
were checked for project account
identifiers, artifact names, personal paths, host addresses, source-history
fingerprints, and file metadata. Required upstream copyright attribution
is retained. See [anonymous release notes](anonymity.md).
