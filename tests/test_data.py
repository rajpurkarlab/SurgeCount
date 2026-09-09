"""Small real parquet fixtures exercise v3 copy, sharding, metadata and labels."""

import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from surgecount.data import add_stop_signal, inspect_dataset, stop_targets, vector_stats
from surgecount.tasks import ALL_CAMERAS, TASK_LABELS


def tree_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def dataset12(tmp_path):
    root = tmp_path / "raw"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/unchanged.mp4").write_bytes(b"synthetic video placeholder - never decoded")
    tasks = np.array([9] * 14 + [8] * 2 + [11] * 12 + [9] * 3 + [0] * 2 + [9] * 3)
    episodes = np.array([0] * 28 + [1] * 8)
    frames = np.r_[np.arange(28), np.arange(8)]
    actions = np.arange(len(tasks) * 12, dtype=np.float32).reshape(-1, 12) / 10
    table = pa.table(
        {
            "episode_index": episodes,
            "frame_index": frames,
            "task_index": tasks,
            "index": np.arange(len(tasks)),
            "timestamp": frames / 10,
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 12)),
            "observation.state": pa.array((actions + 1).tolist(), type=pa.list_(pa.float32(), 12)),
        }
    )
    # This is the schema used by the real HF dataset reader, not just Arrow widths.
    import datasets

    hf_metadata = {"info": {"features": datasets.Features.from_arrow_schema(table.schema).to_dict()}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(hf_metadata).encode()})
    # Deliberately split within a task, then within an episode, in v3 shared files.
    for number, (start, end) in enumerate(((0, 7), (7, 30), (30, len(tasks)))):
        pq.write_table(table.slice(start, end - start), root / f"data/chunk-000/file-{number:03d}.parquet")
    features = {
        "action": {"dtype": "float32", "shape": [12], "names": [f"joint_{i}" for i in range(12)]},
        "observation.state": {"dtype": "float32", "shape": [12]},
        **{camera: {"dtype": "video", "shape": [480, 640, 3]} for camera in ALL_CAMERAS},
    }
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_frames": len(tasks),
        "total_episodes": 2,
        "total_tasks": 13,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    stats = {"action": vector_stats(actions), "observation.state": vector_stats(actions + 1)}
    (root / "meta/stats.json").write_text(json.dumps(stats))
    pd.DataFrame({"task_index": list(TASK_LABELS)}, index=pd.Index(list(TASK_LABELS.values()), name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )
    meta = {
        "episode_index": [0, 1],
        "dataset_from_index": [0, 28],
        "dataset_to_index": [28, 36],
        "length": [28, 8],
        "videos/example/from_timestamp": [0.0, 2.8],
        "videos/example/to_timestamp": [2.8, 3.6],
    }
    for name in stats["action"]:
        meta[f"stats/action/{name}"] = [vector_stats(actions[episodes == ep])[name] for ep in (0, 1)]
    pq.write_table(pa.table(meta), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def test_stop_targets_segments_shards_and_unordered_input():
    # Same task ID across an episode boundary must not be merged.
    episodes = np.array([0] * 6 + [1] * 5)
    frames = np.r_[np.arange(6), np.arange(5)]
    tasks = np.array([9, 9, 9, 0, 9, 9, 9, 9, 9, 8, 8])
    expected = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)
    order = np.array([7, 0, 10, 5, 6, 2, 8, 9, 3, 4, 1])
    np.testing.assert_array_equal(stop_targets(episodes[order], frames[order], tasks[order], 2), expected[order])


@pytest.mark.parametrize("frames", ([0, 0], [0, 2]))
def test_stop_rejects_duplicate_or_missing_frames(frames):
    with pytest.raises(ValueError, match="unique and consecutive"):
        stop_targets([0, 0], frames, [9, 9])


def test_converter_preserves_source_videos_and_episode_offsets(dataset12, tmp_path):
    before = tree_hashes(dataset12)
    destination = tmp_path / "prepared"
    provenance = add_stop_signal(dataset12, destination)
    assert tree_hashes(dataset12) == before
    assert provenance["frames"] == 36
    assert not (destination / "SURGECOUNT_INCOMPLETE").exists()
    assert (destination / "videos/unchanged.mp4").read_bytes() == (dataset12 / "videos/unchanged.mp4").read_bytes()
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [13]
    assert info["features"]["action"]["names"][-1] == "stop_signal"
    tables = [pq.read_table(path) for path in sorted((destination / "data").rglob("*.parquet"))]
    actions = np.asarray(pa.concat_tables(tables)["action"].to_pylist())
    np.testing.assert_array_equal(actions[:4, 12], 0)
    np.testing.assert_array_equal(actions[4:14, 12], 1)  # Segment spans parquet files.
    np.testing.assert_array_equal(actions[14:18, 12], 0)  # Drop and early next task.
    np.testing.assert_array_equal(actions[18:31, 12], 1)
    np.testing.assert_array_equal(actions[31:33, 12], 0)
    np.testing.assert_array_equal(actions[33:, 12], 1)
    stats = json.loads((destination / "meta/stats.json").read_text())
    original_stats = json.loads((dataset12 / "meta/stats.json").read_text())
    assert stats["observation.state"] == original_stats["observation.state"]
    for name, values in vector_stats(actions).items():
        np.testing.assert_allclose(stats["action"][name], values)
    episode_path = "meta/episodes/chunk-000/file-000.parquet"
    original_meta, new_meta = pq.read_table(dataset12 / episode_path), pq.read_table(destination / episode_path)
    for name in original_meta.column_names:
        if not name.startswith("stats/action/"):
            assert original_meta[name].equals(new_meta[name])
    for row, (start, end) in enumerate(((0, 28), (28, 36))):
        for name, values in vector_stats(actions[start:end]).items():
            np.testing.assert_allclose(new_meta[f"stats/action/{name}"][row].as_py(), values)
    # The installed HF parser must accept the resized embedded feature schema.
    import datasets

    loaded = datasets.Dataset.from_parquet(
        [str(path) for path in sorted((destination / "data").rglob("*.parquet"))],
        cache_dir=str(tmp_path / "hf-cache"),
    )
    assert loaded.features["action"].length == 13
    assert len(loaded[0]["action"]) == 13


def test_converter_validates_existing_action13_and_refuses_overwrite(dataset12, tmp_path):
    prepared = tmp_path / "prepared"
    add_stop_signal(dataset12, prepared)
    with pytest.raises(FileExistsError):
        add_stop_signal(dataset12, prepared)
    with pytest.raises(ValueError, match="outside"):
        add_stop_signal(dataset12, dataset12 / "nested")
    second = tmp_path / "second"
    add_stop_signal(prepared, second)
    file = next((prepared / "data").rglob("*.parquet"))
    table = pq.read_table(file)
    values = table["action"].to_pylist()
    values[0][12] = 1 - values[0][12]
    pq.write_table(
        table.set_column(table.column_names.index("action"), "action", pa.array(values, type=table["action"].type)),
        file,
    )
    with pytest.raises(ValueError, match="Existing action13"):
        add_stop_signal(prepared, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_inspect_counts_from_rows_and_checks_task_names(dataset12):
    report = inspect_dataset(dataset12, min_episode_index=1)
    assert report["eligible_training_frames"] == 6
    assert report["selected_episode_indices"] == [1]
    assert report["eligible_frames_per_task"] == {"9": 6}
    tasks = pd.read_parquet(dataset12 / "meta/tasks.parquet")
    tasks = tasks.rename(index={TASK_LABELS[9]: "edited prompt"})
    tasks.to_parquet(dataset12 / "meta/tasks.parquet")
    with pytest.raises(ValueError, match="Task 9"):
        inspect_dataset(dataset12)


def test_canonicalize_raw_first_occurrence_ids_and_done_event(dataset12, tmp_path):
    raw_ids = {9: 0, 8: 1, 11: 2, 0: 3}
    for path in (dataset12 / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        values = [raw_ids[task] for task in table["task_index"].to_pylist()]
        table = table.set_column(
            table.column_names.index("task_index"), "task_index", pa.array(values, type=pa.int64())
        )
        pq.write_table(table, path)
    pd.DataFrame(
        {"task_index": [0, 1, 2, 3]},
        index=pd.Index(
            [
                TASK_LABELS[9],
                TASK_LABELS[8],
                TASK_LABELS[11],
                "subtask finished",
            ],
            name="task",
        ),
    ).to_parquet(dataset12 / "meta/tasks.parquet")
    episode_file = dataset12 / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(episode_file).append_column(
        "tasks", pa.array([["subtask finished"], ["subtask finished"]])
    )
    pq.write_table(metadata, episode_file)
    before = tree_hashes(dataset12)
    with pytest.raises(ValueError, match="Task"):
        add_stop_signal(dataset12, tmp_path / "refused")
    destination = tmp_path / "canonical"
    report = add_stop_signal(dataset12, destination, canonicalize_tasks=True)
    assert report["canonicalized_task_ids"] == {0: 9, 1: 8, 2: 11, 3: 0}
    assert tree_hashes(dataset12) == before
    assert inspect_dataset(destination, min_episode_index=1)["eligible_training_frames"] == 6
    tasks = pd.read_parquet(destination / "meta/tasks.parquet")
    assert tasks.index.tolist() == list(TASK_LABELS.values())
    assert json.loads((destination / "meta/info.json").read_text())["total_tasks"] == 13
    stats = json.loads((destination / "meta/stats.json").read_text())
    assert stats["task_index"]["min"] == [0]
    assert stats["task_index"]["max"] == [11]
    updated = pq.read_table(destination / episode_file.relative_to(dataset12))
    assert "stats/task_index/mean" in updated.column_names
    assert updated["stats/task_index/count"].to_pylist() == [[28], [8]]
    assert updated["tasks"][1].as_py() == [TASK_LABELS[9], TASK_LABELS[0]]
