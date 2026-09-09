# Data and model artifacts

**Datasets and checkpoints will be released upon publication.** This anonymous
review package provides source code and configurations. Project artifact
identifiers and download links will accompany the data and model release.

## Policy and baseline

SurgeCount fully finetunes a language-conditioned policy for eleven skills.
Its thirteen-dimensional output contains twelve robot commands and a stop
signal. The operator or Gemini composition harness supplies the current
subtask instruction and selects the next skill at a stopping boundary.
The reported checkpoint steps are 4,000, 6,000, 9,000, and 10,000.

The end-to-end pi0.5 baseline is implemented from pi0 with a trained subtask
predictor, following the original pi0.5 paper (arXiv:2504.16054). At each action
chunk boundary, the predictor takes the overhead image and an umbrella task
instruction, emits subtask text, and uses that text to condition action
generation. This baseline outputs twelve robot commands without a stop signal.
The predictor-training branch and exact objective remain to be recovered.

The available skill trainer uses LeRobot's `pi05` checkpoint interface.
Supply a compatible initialization checkpoint explicitly; the example leaves
`base_model` unset.

## Using local artifacts

The training configuration pairs `local/gauze_training` with an explicit
local dataset root for a real run. Prepare your own demonstrations with the
[dataset tools](data.md), then follow the
[training guide](training.md). Inspect the action width: twelve-dimensional
recordings need a separate thirteen-dimensional training copy.

Supply local initialization with `surgecount-train --base-model
/path/to/initialization_checkpoint`. The checkpoint's configuration and
weights must load through the release trainer's `pi05` interface.

Deployment accepts a local checkpoint directory with `surgecount-deploy
--checkpoint /path/to/checkpoint`. Required files include `config.json`,
`model.safetensors`, saved processor JSON files, and their normalization
state files. Provision the checkpoint's tokenizer in the target machine's
cache before offline robot operation. Preserve the saved processor transforms
and weights, and match their configuration and source version. Loading
validates weight presence and compatibility.

The historical deployment preset uses renamed OpenPI image keys. New
checkpoints from this release's training wrapper use native camera keys;
set `deployment.image_keys` to `native` for those checkpoints.

For reproducible runs, record immutable dataset and initialization revisions
in your private run configuration. Store run manifests and model outputs
separately from the anonymous source package because they can contain local
paths and artifact identities. The retained deployment launcher selects step
9,000. Mapping each paper result to its evaluation checkpoint requires the
original evaluation records; those mappings and historical artifact revisions
remain unresolved.
