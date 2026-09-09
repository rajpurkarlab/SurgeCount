import json
import pickle
import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from surgecount.tasks import ALL_CAMERAS, TASK_CAMERA, TASK_LABELS
from surgecount.training import (
    TaskFilteredDataset,
    TrainingConfig,
    eligible_row_indices,
    load_config,
    main,
    mask_sample,
    masked_collate,
    resolve_initialization,
    run_training,
    training_hooks,
    validate_training_dataset,
)


def sample(task_id, value=17):
    return {
        "task_index": task_id,
        "task": TASK_LABELS[task_id],
        **{camera: np.full((3, 2, 4), value, dtype=np.uint8) for camera in ALL_CAMERAS},
    }


def test_filter_uses_subset_positions_not_global_offsets():
    assert eligible_row_indices([251, 251, 500, 500], [0, 9, 8, 11]) == [1, 3]
    assert eligible_row_indices([250, 251, 500], [9, 9, 11], max_episode_index=251) == [1]
    with pytest.raises(ValueError, match="No eligible"):
        eligible_row_indices([250, 251], [9, 0])
    with pytest.raises(ValueError, match="Unknown"):
        eligible_row_indices([251], [99])


@pytest.mark.parametrize("task_id", TASK_LABELS)
def test_exact_camera_map_and_nonmutation(task_id):
    original = sample(task_id)
    masked = mask_sample(original)
    for camera in ALL_CAMERAS:
        expected = 17 if TASK_CAMERA.get(task_id) == camera else 0
        assert np.all(masked[camera] == expected)
        assert np.all(original[camera] == 17)
        assert masked[camera].dtype == np.uint8
    assert masked["task"] == TASK_LABELS[task_id]


def test_mask_fails_on_missing_camera_or_wrong_prompt():
    bad = sample(9)
    del bad[ALL_CAMERAS[0]]
    with pytest.raises(ValueError, match="Missing"):
        mask_sample(bad)
    bad = sample(9)
    bad["task"] = "a different skill"
    with pytest.raises(ValueError, match="Instruction"):
        mask_sample(bad)


class FakeDataset:
    def __init__(self):
        self.hf_dataset = SimpleNamespace(
            data=pa.table(
                {
                    "episode_index": [251, 251, 503, 503],
                    "task_index": [0, 9, 8, 11],
                    "index": [9000, 9001, 15000, 15001],
                }
            )
        )
        self.meta = object()
        self.seen = []

    def __getitem__(self, index):
        self.seen.append(index)
        row = sample([0, 9, 8, 11][index])
        # Native action chunks remain intact; only their selected anchor changes.
        row["action"] = np.full((50, 13), index, dtype=np.float32)
        return row

    def __getitems__(self, indices):
        raise AssertionError("Batch optimization must not bypass the filtered wrapper")


def test_wrapper_real_dataloader_sparse_episodes_and_worker_pickle():
    torch = pytest.importorskip("torch")
    base = FakeDataset()
    filtered = TaskFilteredDataset(base)
    assert len(filtered) == 2
    assert filtered.meta is base.meta
    assert filtered.num_episodes == 2
    assert filtered.selection_report["eligible_episode_indices"] == [251, 503]
    batch = next(iter(torch.utils.data.DataLoader(filtered, batch_size=2)))
    assert base.seen == [1, 3]
    assert batch["task_index"].tolist() == [9, 11]
    assert batch["action"].shape == (2, 50, 13)
    assert torch.all(batch["action"][0] == 1)
    assert torch.all(batch[ALL_CAMERAS[0]] == 0)
    assert torch.all(batch[ALL_CAMERAS[2]] == 17)
    restored = pickle.loads(pickle.dumps(filtered))
    assert restored[1]["task_index"] == 11


def test_top_level_collator_tensor_mask_and_pickle():
    torch = pytest.importorskip("torch")
    samples = [sample(2), sample(5)]
    for row in samples:
        for camera in ALL_CAMERAS:
            row[camera] = torch.tensor(row[camera], dtype=torch.float32) / 255
    restored = pickle.loads(pickle.dumps(masked_collate))
    batch = restored(samples)
    assert torch.all(batch[ALL_CAMERAS[0]][0] > 0)
    assert torch.all(batch[ALL_CAMERAS[0]][1] == 0)
    assert torch.all(batch[ALL_CAMERAS[1]][0] == 0)
    assert torch.all(batch[ALL_CAMERAS[1]][1] > 0)


def test_config_and_no_dependency_dry_run(tmp_path, capsys):
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"batch_size": 192, "steps": 10000}))
    loaded = load_config(config)
    assert loaded.min_episode_index == 251
    assert loaded.dataset_repo_id == "local/gauze_training"
    assert loaded.output_dir == "outputs/train/skill_singlecam"
    assert loaded.base_model is None
    main(["--config", str(config), "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["full_finetune"] is True
    assert output["excluded_anchor_task_ids"] == [0, 8]
    assert output["historical_configuration_fully_recovered"] is False
    assert not (tmp_path / "outputs").exists()
    config.write_text('{"typo_batch_size": 192}')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config)
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(batch_size=0).validate()


def block_training_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("lerobot"):
            raise AssertionError("Local-dataset validation must happen before any Hub/LeRobot import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_anonymous_real_run_requires_local_data_before_hub_import(monkeypatch, tmp_path):
    block_training_imports(monkeypatch)
    with pytest.raises(ValueError, match="Datasets and checkpoints will be released upon publication"):
        run_training(TrainingConfig(output_dir=str(tmp_path / "run")))
    assert not (tmp_path / "run").exists()


def test_incomplete_local_tree_fails_before_hub_import(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    block_training_imports(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Datasets and checkpoints will be released upon publication"):
        run_training(TrainingConfig(dataset_root=str(root), output_dir=str(tmp_path / "run")))
    assert not (tmp_path / "run").exists()


def test_anonymous_example_dry_run_and_explicit_dataset_override(monkeypatch, capsys):
    block_training_imports(monkeypatch)
    config = Path(__file__).resolve().parents[1] / "configs/train.json"
    main(["--config", str(config), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["configuration"]["dataset_repo_id"] == "local/gauze_training"
    assert result["configuration"]["base_model"] is None
    assert result["configuration"]["dataset_root"] is None
    owner, repository = "example-user", "my-demonstrations"
    remote_fixture = "/".join((owner, repository))
    explicit = load_config(config, dataset_repo_id=remote_fixture)
    explicit.validate(require_local_data=True)
    assert explicit.dataset_repo_id == remote_fixture


@pytest.fixture
def local_dataset_tree(tmp_path):
    # Only filesystem completeness is needed: initialization must be checked
    # before the dataset parser or other ML imports can run.
    root = tmp_path / "dataset"
    for name in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/file.parquet",
        "data/file.parquet",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return root


@pytest.mark.parametrize("initialization", [None, ""])
def test_missing_initialization_fails_offline(monkeypatch, local_dataset_tree, tmp_path, initialization):
    block_training_imports(monkeypatch)
    with pytest.raises(ValueError, match="explicit initialization checkpoint is required"):
        run_training(
            TrainingConfig(
                dataset_root=str(local_dataset_tree), base_model=initialization, output_dir=str(tmp_path / "run")
            )
        )
    assert not (tmp_path / "run").exists()


def test_missing_relative_checkpoint_never_becomes_hub_request(monkeypatch, local_dataset_tree, tmp_path):
    monkeypatch.chdir(tmp_path)
    block_training_imports(monkeypatch)
    relative_checkpoint = str(Path("weights") / "checkpoint")
    with pytest.raises(FileNotFoundError, match="Local initialization checkpoint is missing"):
        run_training(
            TrainingConfig(
                dataset_root=str(local_dataset_tree), base_model=relative_checkpoint, output_dir=str(tmp_path / "run")
            )
        )
    with pytest.raises(FileNotFoundError, match="Local initialization checkpoint is missing"):
        resolve_initialization(str(tmp_path / "missing"))


def test_invalid_initialization_config_fails_offline(monkeypatch, local_dataset_tree, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weight loading is outside this configuration test")
    (checkpoint / "config.json").write_text("not json")
    block_training_imports(monkeypatch)
    config = TrainingConfig(
        dataset_root=str(local_dataset_tree), base_model=str(checkpoint), output_dir=str(tmp_path / "run")
    )
    with pytest.raises(ValueError, match="Invalid initialization checkpoint config"):
        run_training(config)
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi0"}))
    with pytest.raises(ValueError, match="Expected checkpoint policy type pi05"):
        run_training(config)


def test_cli_explicit_checkpoint_and_intentional_hub_initialization(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi05"}))
    (checkpoint / "model.safetensors").write_bytes(b"weight loading is outside this configuration test")
    block_training_imports(monkeypatch)
    main(["--base-model", str(checkpoint), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["configuration"]["base_model"] == str(checkpoint)
    assert resolve_initialization(str(checkpoint)) == ("local", str(checkpoint.resolve()))
    owner, repository = "example-user", "initialization"
    remote_id = "/".join((owner, repository))
    assert resolve_initialization("hf://" + remote_id) == ("hub", remote_id)
    with pytest.raises(ValueError, match="valid namespace and repository"):
        resolve_initialization("hf://" + owner)


def test_scoped_processors_use_new_stats_and_hooks_restore():
    seen = {}

    def processors(**kwargs):
        seen.update(kwargs)
        return "pre", "post"

    module = SimpleNamespace(
        make_dataset=lambda cfg: None,
        make_pre_post_processors=processors,
        make_policy=lambda **kwargs: SimpleNamespace(named_parameters=lambda: []),
    )
    original_policy = module.make_policy
    stats = {"action": {"mean": [0] * 13}}
    with training_hooks(module, TrainingConfig()):
        assert module.make_pre_post_processors(
            "pi05config", "base12", dataset_stats=stats, preprocessor_overrides={"old": "shape"}
        ) == ("pre", "post")
        assert seen == {"policy_cfg": "pi05config", "pretrained_path": None, "dataset_stats": stats}
    assert module.make_pre_post_processors is processors
    assert module.make_policy is original_policy


def test_full_finetuning_rejects_frozen_parameters():
    module = SimpleNamespace(
        make_dataset=lambda cfg: None,
        make_pre_post_processors=lambda **kwargs: None,
        make_policy=lambda **kwargs: SimpleNamespace(
            named_parameters=lambda: [("vision", SimpleNamespace(requires_grad=False))]
        ),
    )
    with training_hooks(module, TrainingConfig()):
        with pytest.raises(RuntimeError, match="frozen"):
            module.make_policy()


def test_dataset_preflight_checks_all_stop_labels_and_normalization():
    from surgecount.data import vector_stats

    # Long enough segment contains both stop values and a nondegenerate range.
    actions = np.ones((20, 13), dtype=np.float32)
    actions[:10, 12] = 0
    metadata = SimpleNamespace(
        features={
            "action": {"shape": [13]},
            "observation.state": {"shape": [12]},
            **{camera: {"dtype": "video"} for camera in ALL_CAMERAS},
        },
        stats={"action": vector_stats(actions), "observation.state": vector_stats(np.ones((20, 12)))},
        tasks=pd.DataFrame({"task_index": list(TASK_LABELS)}, index=list(TASK_LABELS.values())),
    )
    table = pa.table(
        {
            "episode_index": [251] * 20,
            "frame_index": list(range(20)),
            "task_index": [9] * 20,
            "action": actions.tolist(),
        }
    )
    dataset = SimpleNamespace(meta=metadata, hf_dataset=SimpleNamespace(data=table))
    validate_training_dataset(dataset)
    metadata.stats["action"]["q99"] = [1] * 12
    with pytest.raises(ValueError, match="normalization"):
        validate_training_dataset(dataset)
    metadata.stats["action"] = vector_stats(actions)
    actions[0, 12] = 1
    dataset.hf_dataset.data = table.set_column(table.column_names.index("action"), "action", pa.array(actions.tolist()))
    with pytest.raises(ValueError, match="Stop targets"):
        validate_training_dataset(dataset)
