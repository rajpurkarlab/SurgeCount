# Source provenance and scope

This release was assembled from the preserved robot-side LeRobot source,
the separate inference-workstation source bundle, and the existing static
project website. Historical memory and surviving operator prompts were
used to interpret the code, not copied into the public repository.

The LeRobot snapshot reports version 0.5.2 and includes local research
modifications. Private source history and commit identifiers are withheld
for anonymous review; the vendored files define this release's source tree.
The newer workstation recorder, control utilities, Gemini planner, and
Linux keyboard support were integrated into that snapshot. See
`vendor/lerobot/SURGECOUNT_CHANGES.md` for the changed components and
compatibility fixes. Upstream copyright and license notices are retained.

The release adds standalone configuration-driven commands, fresh-copy
dataset preparation, robust dataset-position filtering, camera masking,
failure checks for pretrained loading, and offline regression tests.
The old experimental shell launcher collection is not the public interface.
Model weights, calibration, home poses, device serials, private machine
paths, API keys, original chat histories, and raw datasets are excluded.

## Experimental interpretation

The confirmed final SurgeCount recipe uses a language-conditioned policy,
the curated demonstrations after episode 250, 13-dimensional actions, idle/drop anchor
filtering, single-camera masking, full finetuning, and batch 192. The
recovered workstation launcher defaults to step 9,000 and a five-prediction
lookahead stop window. The earlier manual launcher used a ten-control-step
debounce. Both stop rules are documented and selectable; their settings
are not conflated.

The recovered end-to-end pi0.5 baseline is implemented from pi0 with a trained
subtask predictor, following the original pi0.5 paper (arXiv:2504.16054).
The overhead image and umbrella instruction feed the predictor at each action
chunk boundary; its subtask text then conditions twelve-dimensional action
generation. This baseline has no stop signal. SurgeCount uses the separate
thirteen-dimensional skill policy and operator/Gemini composition harness.
The baseline predictor-training branch and exact objective remain pending
source recovery.

The release trainer retains LeRobot's `pi05` checkpoint interface. Its
initialization checkpoint is supplied explicitly with `--base-model`;
the example configuration leaves `base_model` unset.

The manuscript is maintained separately. Its policy labels, dataset
counts, and stop description are not treated as a replacement for the
confirmed code recipe. This release does not reconstruct evaluation labels,
per-trial outcome tables, historical base/dataset revision hashes, or an
exact deleted training launch. Those artifacts were not supplied.

The data curation is documented as recovered history.
Where exact episode boundaries or task shorthand are unresolved, the code
does not guess and silently reconstruct the final dataset. It
accepts the curated artifact or prepares a new dataset supplied by an
operator using explicit task strings and stop-label rules.

Datasets and checkpoints will be released upon publication.
