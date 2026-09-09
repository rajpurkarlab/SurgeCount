# Release validation

Release validation covers offline software, packaging, and website checks.

## Checks performed

- A fresh, isolated Python3.12 environment was resolved from `uv.lock` and
  installed with the data/development extras. The lock resolves PyTorch2.10.0
  for the optional full environment and pins transformers5.3.0.
- The collection, training, and deployment console scripts passed dry runs
  from an ordinary package installation. The data console script exposes
  both audit and fresh-copy preparation commands.
- 106 offline tests passed in the available test environment with PyTorch;
  ten additional unittest subtests passed. The lightweight data environment
  skips the two PyTorch-dependent cases.
- The suite covers real Parquet/Arrow metadata, shard-spanning stop labels,
  source-file hash preservation, canonical task remapping, action13 global
  and per-episode statistics, sparse episode indexing, camera masking,
  dataloader behavior, and spawn-compatible collate functions.
- Runtime checks cover offline configuration, calibration/checkpoint
  preflight, stop lookahead, interpolation, failed checkpoint loading,
  dynamic action-queue resizing, and operator priority over late planner
  responses. These checks use fixtures and stubs.
- All vendored Python files parse under Python3.12. The recorder and trainer
  import using the existing robot environment with transformers5.3.0.
  Generated recording/deployment arguments parse through the actual LeRobot
  dataclasses. A training-config integration check uses mocked metadata and
  weight-loading entry points.
- First-party lint, release-content scanning, website local-target checks,
  and image/video metadata checks pass. Root and vendored packages build.
- Anonymity regressions cover account links, personal paths, hidden metadata,
  binary identifier scans, and exact ZIP membership and neutral metadata.
  Local training placeholders cannot trigger a remote dataset lookup.
  Training requires explicit initialization; missing or incompatible local
  checkpoints fail before ML imports or Hub access. Remote initialization
  requires an explicit Hub URI.
  Video metadata checks accept the neutral zero vendor field emitted by
  older FFmpeg versions and continue to reject identifying vendor values.
- The page was inspected in a browser at wide and narrow viewport sizes.
  All eight videos decode successfully. The supplied clip video bitstreams
  are unchanged; container metadata and timecode tracks were stripped.

## Deployment validation

Training, real checkpoint loading, GPU inference, physical robot operation,
and live planner calls remain untested in this release audit. Validate these
on the target setup, including native device libraries and GPU wheels.
The full train/robot dependency lock was resolved; the fresh installation
test used the data/development extras.

The deployment preset targets CUDA. MPS is configurable and remains untested.

The example configuration defines defaults and enforces dimensional, task,
and camera consistency. Dry runs validate configuration; checkpoint checks
inspect local metadata. Historical configuration gaps are recorded in
[source provenance](provenance.md).

## Media and artifact review

Visual review covered six frames per clip, all posters, and the architecture
figure, including its training inset with a partial operator profile.
Source and media metadata scans checked the complete source archive and
website for project account identifiers, artifact names, personal paths,
host addresses, source-history fingerprints, and identifying file metadata.
The export allowlist defines the distributed files; upstream copyright
attribution is retained. See [anonymous release notes](anonymity.md).

Datasets and checkpoints will be released upon publication.
