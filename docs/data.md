# Demonstrations and dataset preparation

**Datasets and checkpoints will be released upon publication.** The final dataset contains curated bimanual gauze-handling demonstrations collected by teleoperating two SO-101 arms. The model is conditioned on the instruction for the skill active at each recorded frame. Project account names, repository identifiers, and download links are withheld for anonymous review.

Compute usable episode IDs and eligible frame counts from the selected local dataset's parquet files. Curation and episode/task filtering determine the training subset.

## Recorded signals

| Signal | Representation |
| --- | --- |
| Left front, overhead, right front views | Three 640×480 camera streams; dataset sampled at 10 Hz |
| `observation.state` | 12 joint/gripper positions, in dataset feature-name order |
| Raw `action` | 12 commanded joint/gripper positions |
| Prepared `action` | Same 12 commands, followed by `stop_signal` at index 12 |
| `task_index` | Canonical integer skill ID; resolved to the exact language instruction |
| Episode/frame/index/timestamp | Native LeRobot temporal and video alignment fields |

Camera acquisition can run at 30 Hz while the dataset records at 10 Hz. Preserve these separate rates and the ordered action/state feature names, including the left/right joint order, during conversion.

The canonical task table and camera selection are defined in `src/surgecount/tasks.py`. Dataset IDs differ from recording keyboard digits. In particular, keyboard `1` records dataset task 9, and keyboard `6` records dataset task 11.

| Dataset task IDs | Active camera | Skill |
| --- | --- | --- |
| 1 | Overhead | Return left arm |
| 2, 4, 12 | Left front | Left release; left release/return; grasp white edge |
| 3, 7, 9, 11 | Right front | Right return; right release; nearest/exposed blue-thread grasp |
| 5, 6, 10 | Overhead | Place in slot 3, 2, 1 respectively |
| 0, 8 | None; excluded training anchors | Idle and failed/drop marker |

## Lineage and curation

The collection grew through repeated recording, cleaning, and merging passes. Recordings after the camera replacement around episode 251 formed the later training subset. Historical artifact names are withheld during anonymous review.

Cleaning removed corrupted recordings and relabeled unwanted, failed, or superseded skill segments as idle. Later additions retained selected skills; some older demonstrations were subsequently retired. The complete append, deletion, and reindexing manifest remains unavailable. The curated dataset will accompany publication; the tools below prepare independently collected local data.

## Inspect a local cleaned tree

Install the data extra as described in the main README, then run:

```bash
surgecount-data inspect /path/to/cleaned_dataset --min-episode-index 251
```

This read-only, offline command reads scalar parquet columns, checks the canonical task mapping, and prints frame counts by task, eligible episode IDs, and the number of eligible training anchors. Inspect the labels and replay representative episodes before training.

Episode bounds refer to **zero-based `episode_index` values, inclusively**. The release uses `episode_index >= 251` for “after 250.” The historical index-versus-ordinal convention is unresolved; a one-based ordinal boundary would select index 250. Set the boundary explicitly for the selected dataset.

## Prepare the stop dimension in a new copy

```bash
surgecount-data add-stop /path/to/cleaned_dataset /path/to/prepared_dataset
```

Use a new destination outside the source tree. The converter creates independent file copies, preserving video bytes, row IDs, timestamps, episode boundaries, and video offsets. Budget storage for a full additional dataset copy.

For each contiguous task segment within an episode, the final ten frames receive stop=1. Shorter segments receive stop=1 throughout. Idle and drop frames always receive stop=0. A task that recurs after another task starts a new segment. Segments that cross v3 parquet shard boundaries are handled together. Adjacent same-task segments without an intervening task/event marker cannot be distinguished from these fields and are treated as one segment.

The converter supports both 12-dimensional input and already prepared 13-dimensional input. Existing stop labels must match the requested rule; inconsistent labels fail validation. It updates:

- The action column and its Arrow/Hugging Face feature schema in every data shard.
- `meta/info.json` action shape and names.
- Global action statistics in `meta/stats.json` and action statistics in every `meta/episodes` parquet shard, including quantiles and count.
- A preparation report in `meta/surgecount_preparation.json`.

Normalization statistics use exact quantiles and population standard deviation over all frames in the copied dataset, including frames excluded from training anchors. Other feature statistics are preserved. Exact quantiles can differ slightly from the approximate histogram quantiles used by historical LeRobot utilities.

If conversion fails after writing begins, the new destination retains a `SURGECOUNT_INCOMPLETE` marker and must not be used for training. Inspect the error and choose a new destination for a corrected run. The original recording remains intact.

## Older recordings with first-occurrence task IDs

LeRobot normally assigns new task IDs by first occurrence. The release's recording wrapper pre-registers the canonical table, but older raw recordings can have a different ID order. Canonicalize them while preparing a new copy:

```bash
surgecount-data add-stop /path/to/raw /path/to/prepared --canonicalize-tasks
```

This flag maps IDs by the **exact instruction string** and maps the legacy `subtask finished` event to idle. It rewrites the copied task table, frame task IDs, episode task lists, and global/episode task-index statistics. Unknown instructions cause an error. Curate skill segments separately from this schema conversion: the collection code relabels failed segments at recording time, while older raw drop markers require review of the preceding skill.

Prepared datasets remain local. Keep source data, derived data, and training outputs in separate directories.
