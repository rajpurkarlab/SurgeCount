# Data and model artifacts

**Datasets and checkpoints will be released upon publication.** This anonymous
review release contains no project Hugging Face account names, artifact
repository identifiers, download links, raw training demonstrations, or
model weights.

The reported policy was fully finetuned from the public upstream
`lerobot/pi05_base` model. That upstream identifier describes the method's
initialization and does not identify the submission authors. The reported
checkpoint steps are 4,000, 6,000, 9,000, and 10,000; project artifact links
will be added upon publication.

## Using local artifacts

The training configuration uses `local/gauze_training` as a placeholder
and requires an explicit local dataset root for a real run. Prepare your
own demonstrations with the [dataset tools](data.md), then follow the
[training guide](training.md). Inspect the action width: twelve-dimensional
recordings need a separate thirteen-dimensional training copy.

Deployment accepts a local checkpoint directory with `surgecount-deploy
--checkpoint /path/to/checkpoint`. Required files include `config.json`,
`model.safetensors`, saved processor JSON files, and their normalization
state files. Provision the tokenizer named by the checkpoint in the target
machine's cache before offline robot operation. Do not strip processor
transforms or modify weights to make a mismatched checkpoint load; resolve
its configuration and source version first. Loading fails on incompatible
or missing weights.

The historical deployment preset uses renamed OpenPI image keys. New
checkpoints from this release's training wrapper use native camera keys;
set `deployment.image_keys` to `native` for those checkpoints.

For reproducible future runs, record immutable dataset and base-model
revisions in your private run configuration. Keep run manifests and model
outputs outside the anonymous source release: they can record local paths
and artifact identities. No missing historical revision or evaluation
checkpoint selection is invented here. A retained launch setting used
step 9,000; that alone does not establish which checkpoint produced every
paper result.
