"""Read-only dataset audit and copy-only LeRobot v3 stop-target preparation.

Only scalar columns and actions are loaded into memory. Videos are copied as
bytes, and source recordings are never opened for writing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from surgecount.tasks import EXCLUDE_TASKS, TASK_LABELS, validate_task_labels


def _deps():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    return np, pa, pq


def stop_targets(episode_indices, frame_indices, task_indices, last_n=10):
    """Return binary targets aligned to input order, including across shards.

    Every contiguous non-idle task segment has its last min(last_n, length)
    frames marked complete. A task repeated after another task is a new segment.
    Episode boundaries always split segments, even for identical task IDs.
    """
    np, _, _ = _deps()
    if not isinstance(last_n, int) or isinstance(last_n, bool) or last_n < 1:
        raise ValueError("last_n must be a positive integer")
    episodes = np.asarray(episode_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    tasks = np.asarray(task_indices, dtype=np.int64)
    if not (episodes.ndim == frames.ndim == tasks.ndim == 1):
        raise ValueError("episode_index, frame_index, task_index must be one-dimensional")
    if not (len(episodes) == len(frames) == len(tasks)) or not len(tasks):
        raise ValueError("Frame columns must be nonempty and equal in length")
    if (episodes < 0).any() or (frames < 0).any():
        raise ValueError("Episode and frame indices must be nonnegative")
    order = np.lexsort((frames, episodes))
    ep, fr, task = episodes[order], frames[order], tasks[order]
    same_ep = ep[1:] == ep[:-1]
    if np.any(fr[np.r_[True, ~same_ep]] != 0):
        raise ValueError("Each complete episode must start at frame_index 0")
    if np.any(same_ep & (np.diff(fr) != 1)):
        raise ValueError("Frames within each episode must be unique and consecutive")
    boundaries = np.r_[0, np.flatnonzero((ep[1:] != ep[:-1]) | (task[1:] != task[:-1])) + 1, len(task)]
    result = np.zeros(len(task), dtype=np.float32)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if int(task[start]) not in EXCLUDE_TASKS:
            result[order[max(start, end - last_n) : end]] = 1.0
    return result


def vector_stats(values):
    """Exact vector statistics, with LeRobot's population std/count convention."""
    np, _, _ = _deps()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics need a nonempty finite matrix")
    result = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        result[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return result


def read_frame_columns(root, include_actions=False):
    """Read physical row order; it is deliberately never replaced by global index."""
    np, _, pq = _deps()
    root = Path(root)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files in {root}")
    names = ["episode_index", "frame_index", "task_index", "index"]
    if include_actions:
        names.append("action")
    chunks = {name: [] for name in names}
    sizes = []
    for file in files:
        table = pq.read_table(file, columns=names)
        sizes.append(len(table))
        for name in names:
            values = table[name].to_pylist() if name == "action" else table[name].to_numpy()
            chunks[name].append(np.asarray(values))
    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    if not len(arrays["index"]):
        raise ValueError("Dataset is empty")
    if len(np.unique(arrays["index"])) != len(arrays["index"]):
        raise ValueError("Duplicate global frame indices")
    return files, sizes, arrays


def task_labels_from_root(root):
    _, _, pq = _deps()
    tasks = pq.read_table(Path(root) / "meta/tasks.parquet").to_pandas()
    # LeRobot stores the instruction as a pandas index and task_index as a column.
    if "task_index" not in tasks:
        raise ValueError("meta/tasks.parquet is missing task_index")
    labels = {int(row["task_index"]): str(label) for label, row in tasks.iterrows()}
    if list(map(int, tasks["task_index"])) != list(range(len(tasks))):
        raise ValueError("Task metadata rows must be ordered by contiguous task_index")
    return labels


def inspect_dataset(root, min_episode_index=251, max_episode_index=None):
    np, _, _ = _deps()
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    _, _, columns = read_frame_columns(root)
    validate_task_labels(task_labels_from_root(root), np.unique(columns["task_index"]))
    # Also validate frame order/duplicates; do not derive targets or mutate data.
    stop_targets(columns["episode_index"], columns["frame_index"], columns["task_index"])
    selected = (columns["episode_index"] >= min_episode_index) & ~np.isin(columns["task_index"], list(EXCLUDE_TASKS))
    if max_episode_index is not None:
        if max_episode_index < min_episode_index:
            raise ValueError("max_episode_index must be >= min_episode_index")
        selected &= columns["episode_index"] <= max_episode_index
    count = lambda values: {str(k): int(v) for k, v in sorted(Counter(map(int, values)).items())}
    return {
        "root": str(root),
        "format": info.get("codebase_version"),
        "fps": info.get("fps"),
        "action_shape": info.get("features", {}).get("action", {}).get("shape"),
        "total_frames_observed": len(columns["index"]),
        "total_episodes_observed": len(np.unique(columns["episode_index"])),
        "min_episode_index_inclusive": min_episode_index,
        "max_episode_index_inclusive": max_episode_index,
        "selected_episode_indices": sorted(map(int, np.unique(columns["episode_index"][selected]))),
        "eligible_training_frames": int(selected.sum()),
        "all_frames_per_task": count(columns["task_index"]),
        "eligible_frames_per_task": count(columns["task_index"][selected]),
    }


def _replace_vector_column(table, name, values, dtype):
    """Update both Arrow and embedded Hugging Face feature metadata."""
    _, pa, _ = _deps()
    length = len(values[0])
    array = pa.array(values, type=pa.list_(dtype, length))
    if name in table.column_names:
        table = table.set_column(table.column_names.index(name), name, array)
    else:
        table = table.append_column(name, array)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        features = hf.get("info", {}).get("features")
        if features is not None:
            features[name] = {
                "feature": {
                    "dtype": {"float": "float32", "double": "float64"}.get(str(dtype), str(dtype)),
                    "_type": "Value",
                },
                "length": length,
                "_type": "List",
            }
            metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def add_stop_signal(source, destination, last_n=10, canonicalize_tasks=False):
    """Make a complete new v3 tree with action13 and refreshed action statistics.

    Existing action13 labels are validated against the requested stop rule.
    A failed write leaves an INCOMPLETE marker in the new destination.
    Source recordings remain read-only throughout conversion.
    """
    np, pa, pq = _deps()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Source contains symlinks; materialize a standalone dataset copy first")
    info = json.loads((source / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("This converter supports LeRobot v3 datasets only")
    files, sizes, columns = read_frame_columns(source, include_actions=True)
    actions = np.asarray(columns["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] not in (12, 13) or not np.isfinite(actions).all():
        raise ValueError("action must contain exactly 12 or 13 finite values per frame")
    if info["features"]["action"]["shape"] != [actions.shape[1]]:
        raise ValueError("Action parquet width does not match meta/info.json")
    if len(actions) != info["total_frames"]:
        raise ValueError("Observed frame count disagrees with meta/info.json")
    source_labels = task_labels_from_root(source)
    task_mapping = None
    if canonicalize_tasks:
        name_to_id = {label: task_id for task_id, label in TASK_LABELS.items()}
        # The recording completion key is an event marker, never a policy skill.
        name_to_id["subtask finished"] = 0
        unknown_labels = set(source_labels.values()) - set(name_to_id)
        if unknown_labels:
            raise ValueError(f"Cannot canonicalize unknown instructions: {sorted(unknown_labels)}")
        task_mapping = {old: name_to_id[label] for old, label in source_labels.items()}
        missing_ids = set(map(int, np.unique(columns["task_index"]))) - set(task_mapping)
        if missing_ids:
            raise ValueError(f"Frame task IDs are absent from task metadata: {sorted(missing_ids)}")
        columns["task_index"] = np.array([task_mapping[int(task)] for task in columns["task_index"]], dtype=np.int64)
    else:
        validate_task_labels(source_labels, np.unique(columns["task_index"]))
    targets = stop_targets(columns["episode_index"], columns["frame_index"], columns["task_index"], last_n)
    if actions.shape[1] == 13:
        if not np.array_equal(actions[:, 12], targets):
            raise ValueError("Existing action13 stop labels do not match contiguous-segment targets")
        output_actions = actions
    else:
        output_actions = np.column_stack((actions, targets))
    episode_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError("LeRobot v3 episode metadata parquet files are required")
    episode_tables = [(file, pq.read_table(file)) for file in episode_files]
    meta_ids = [int(ep) for _, table in episode_tables for ep in table["episode_index"].to_pylist()]
    observed_ids = set(map(int, np.unique(columns["episode_index"])))
    if len(meta_ids) != len(set(meta_ids)) or set(meta_ids) != observed_ids:
        raise ValueError("Episode metadata does not match data episode IDs exactly")
    if len(meta_ids) != info["total_episodes"]:
        raise ValueError("Observed episode count disagrees with meta/info.json")
    # The vendored v3 reader indexes metadata rows directly by episode_index.
    if meta_ids != list(range(len(meta_ids))):
        raise ValueError("LeRobot v3 episode metadata must be ordered by contiguous episode_index")
    for _, table in episode_tables:
        for row in table.select(["episode_index", "dataset_from_index", "dataset_to_index", "length"]).to_pylist():
            rows = np.flatnonzero(columns["episode_index"] == int(row["episode_index"]))
            rows = rows[np.argsort(columns["frame_index"][rows])]
            expected = np.arange(int(row["dataset_from_index"]), int(row["dataset_to_index"]))
            if len(rows) != int(row["length"]) or not np.array_equal(columns["index"][rows], expected):
                raise ValueError(f"Episode {row['episode_index']} global indices do not match metadata boundaries")
    by_episode = {ep: vector_stats(output_actions[columns["episode_index"] == ep]) for ep in meta_ids}
    stats = json.loads((source / "meta/stats.json").read_text())
    stats["action"] = vector_stats(output_actions)
    task_stats_by_episode = None
    if canonicalize_tasks:
        stats["task_index"] = vector_stats(columns["task_index"][:, None])
        task_stats_by_episode = {
            ep: vector_stats(columns["task_index"][columns["episode_index"] == ep, None]) for ep in meta_ids
        }
        info["total_tasks"] = len(TASK_LABELS)
    feature = info["features"]["action"]
    feature["shape"] = [13]
    feature["dtype"] = "float32"
    names = feature.get("names")
    if names is not None:
        if not isinstance(names, list) or len(names) not in (12, 13):
            raise ValueError("Action feature names must contain 12 joint names, plus optional stop_signal")
        feature["names"] = names[:12] + ["stop_signal"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive creation; source can never be the write target.
    marker = destination / "SURGECOUNT_INCOMPLETE"
    marker.write_text("Dataset conversion has not completed. Do not train on this directory.\n")
    # No hardlinks: modifying this destination later cannot mutate raw recordings.
    shutil.copytree(source, destination, dirs_exist_ok=True)
    offset = 0
    for file, size in zip(files, sizes, strict=True):
        table = pq.read_table(file)
        table = _replace_vector_column(table, "action", output_actions[offset : offset + size].tolist(), pa.float32())
        if canonicalize_tasks:
            column_index = table.column_names.index("task_index")
            table = table.set_column(
                column_index,
                table.schema.field("task_index"),
                pa.array(columns["task_index"][offset : offset + size], type=table["task_index"].type),
            )
        pq.write_table(table, destination / file.relative_to(source))
        offset += size
    for file, table in episode_tables:
        ids = list(map(int, table["episode_index"].to_pylist()))
        updated_features = {"action": by_episode}
        if canonicalize_tasks:
            updated_features["task_index"] = task_stats_by_episode
        for feature_name, episode_stats in updated_features.items():
            stale = [name for name in table.column_names if name.startswith(f"stats/{feature_name}/")]
            # Replacing existing columns keeps episode/video offsets untouched.
            for stat in stats[feature_name]:
                table = _replace_vector_column(
                    table,
                    f"stats/{feature_name}/{stat}",
                    [episode_stats[ep][stat] for ep in ids],
                    pa.int64() if stat == "count" else pa.float64(),
                )
            for name in stale:
                if name.split("/")[-1] not in stats[feature_name]:
                    raise ValueError(f"Unsupported source statistic {name}; conversion marked incomplete")
        if canonicalize_tasks and "tasks" in table.column_names:
            values = []
            for ep in ids:
                rows = np.flatnonzero(columns["episode_index"] == ep)
                rows = rows[np.argsort(columns["frame_index"][rows])]
                values.append(list(dict.fromkeys(TASK_LABELS[int(task)] for task in columns["task_index"][rows])))
            column_index = table.column_names.index("tasks")
            table = table.set_column(
                column_index, table.schema.field("tasks"), pa.array(values, type=table["tasks"].type)
            )
        pq.write_table(table, destination / file.relative_to(source))
    if canonicalize_tasks:
        import pandas as pd

        pd.DataFrame(
            {"task_index": list(TASK_LABELS)}, index=pd.Index(list(TASK_LABELS.values()), name="task")
        ).to_parquet(destination / "meta/tasks.parquet")
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (destination / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    provenance = {
        "operation": "append_or_validate_stop_signal",
        "source_action_dim": actions.shape[1],
        "stop_last_n_frames": last_n,
        "excluded_stop_task_ids": sorted(EXCLUDE_TASKS),
        "frames": len(actions),
        "episodes": len(meta_ids),
        "action_stats": "exact, all frames; population standard deviation",
        "source_modified": False,
        "canonicalized_task_ids": task_mapping,
    }
    (destination / "meta/surgecount_preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    marker.unlink()
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="Count frames/tasks/episodes without decoding videos")
    inspect.add_argument("root", type=Path)
    inspect.add_argument("--min-episode-index", type=int, default=251)
    inspect.add_argument("--max-episode-index", type=int)
    convert = subparsers.add_parser("add-stop", help="Prepare action13 in a NEW dataset directory")
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--last-n", type=int, default=10)
    convert.add_argument(
        "--canonicalize-tasks",
        action="store_true",
        help="Map first-occurrence raw IDs by exact prompt; completion event becomes idle",
    )
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_dataset(args.root, args.min_episode_index, args.max_episode_index)
    else:
        result = add_stop_signal(args.source, args.destination, args.last_n, args.canonicalize_tasks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
