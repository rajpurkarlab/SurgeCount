# Source provenance

This release combines the preserved robot-side LeRobot source, the
inference-workstation source bundle, and the static project website.
Historical memory and surviving operator prompts informed interpretation of
the code and recipe reconstruction. Those records remain private.

The vendored LeRobot 0.5.2 snapshot includes research modifications and the
newer workstation recorder, control utilities, Gemini planner, and Linux
keyboard support. The vendored files define the distributed source tree;
private source history and commit identifiers are withheld for anonymous
review. See `vendor/lerobot/SURGECOUNT_CHANGES.md` for the changed components and
compatibility fixes. Upstream copyright and license notices are retained.

The public interface consists of standalone configuration-driven commands
for collection, dataset preparation, training, and deployment. The release
adds dataset preparation in new copies, filtering by local row position, camera masking,
strict initialization loading, and offline regression tests. Model weights,
raw datasets, calibration, home poses, device identities, machine paths,
API keys, and original chat histories remain outside the anonymous package.

## Experimental interpretation

The confirmed SurgeCount recipe uses curated demonstrations after episode
250, a language-conditioned policy, 13-dimensional actions, idle/drop anchor
filtering, single-camera masking, full finetuning, and batch 192. Two
selectable runtime profiles preserve the recorded stop rules: the workstation
launcher uses step 9,000 with a five-prediction lookahead window; the earlier
manual launcher uses a ten-control-step debounce.

The end-to-end pi0.5 baseline is implemented from pi0 with a trained
subtask predictor, following the original pi0.5 paper (arXiv:2504.16054).
The overhead image and umbrella instruction feed the predictor at each action
chunk boundary; its subtask text then conditions twelve-dimensional action
generation without a stop signal.
SurgeCount uses the thirteen-dimensional skill policy and operator/Gemini
composition harness. The baseline predictor-training branch and exact objective
remain to be recovered.

The release trainer retains LeRobot's `pi05` checkpoint interface. Its
initialization checkpoint is supplied explicitly with `--base-model`;
the example configuration leaves `base_model` unset.

The manuscript is maintained separately. Runtime settings here follow the
confirmed code recipe. `configs/train.json` identifies reconstruction
defaults; [training.md](training.md) explains their selection. Exact
experimental reproduction also requires the original final training launch,
historical initialization and dataset revision hashes, evaluation labels,
and per-trial outcome tables. These records remain unavailable.

Data preparation accepts the curated artifact or creates a new dataset from
operator-supplied recordings using explicit task strings and stop-label rules.
The recovered curation history records unresolved episode boundaries and
task shorthand; reproducing the historical curated dataset requires those
decisions.

Datasets and checkpoints will be released upon publication.
