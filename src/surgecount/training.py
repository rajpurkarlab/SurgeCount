"""Train a language-conditioned skill policy with local-row filtering and camera masks.

The heavy LeRobot/PyTorch stack is imported only when training starts. --dry-run
can inspect configuration without a GPU, hardware access, or Hub requests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import json
import logging
from pathlib import Path
import re
from unittest.mock import patch

from surgecount.tasks import ALL_CAMERAS, EXCLUDE_TASKS, TASK_CAMERA, TASK_LABELS, validate_task_labels

ARTIFACT_AVAILABILITY = "Datasets and checkpoints will be released upon publication."


@dataclass
class TrainingConfig:
    dataset_repo_id: str = "local/gauze_training"
    dataset_root: str | None = None
    dataset_revision: str | None = None
    base_model: str | None = None
    base_revision: str | None = None
    output_dir: str = "outputs/train/skill_singlecam"
    min_episode_index: int = 251
    max_episode_index: int | None = None
    batch_size: int = 192
    steps: int = 10000
    save_freq: int = 1000
    seed: int = 1000
    num_workers: int = 4
    video_backend: str = "pyav"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    device: str = "cuda"
    stop_last_n_frames: int = 10
    # These are explicit release defaults based on the vendored PI05 preset.
    # They are not asserted to be the deleted historical launch configuration.
    chunk_size: int = 50
    optimizer_lr: float = 2.5e-5
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def validate(self, require_local_data=False):
        for key in ("batch_size", "steps", "save_freq", "stop_last_n_frames", "chunk_size", "scheduler_decay_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("min_episode_index", "num_workers", "seed", "scheduler_warmup_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.max_episode_index is not None:
            if (
                not isinstance(self.max_episode_index, int)
                or isinstance(self.max_episode_index, bool)
                or self.max_episode_index < self.min_episode_index
            ):
                raise ValueError("max_episode_index must be an integer >= min_episode_index")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if not self.dataset_repo_id or not self.output_dir:
            raise ValueError("dataset_repo_id and output_dir are required")
        if self.base_model is not None and not isinstance(self.base_model, str):
            raise ValueError("base_model must be an explicit checkpoint path, Hub URI, or null for a dry run")
        if self.optimizer_lr <= 0 or self.scheduler_decay_lr < 0 or self.optimizer_weight_decay < 0:
            raise ValueError("Invalid optimizer or scheduler values")
        if require_local_data and self.dataset_repo_id.startswith("local/"):
            if not self.dataset_root:
                raise ValueError(
                    f"{ARTIFACT_AVAILABILITY} Supply --dataset-root for a prepared local dataset; "
                    "local/* dataset identities are never remote repository defaults."
                )
            root = Path(self.dataset_root)
            missing = [
                name
                for name in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet")
                if not (root / name).is_file()
            ]
            for directory in ("meta/episodes", "data"):
                if not any((root / directory).rglob("*.parquet")):
                    missing.append(f"{directory}/**/*.parquet")
            if missing:
                raise FileNotFoundError(
                    f"{ARTIFACT_AVAILABILITY} The supplied local dataset is incomplete: {', '.join(missing)}"
                )
        if self.dataset_root and not (Path(self.dataset_root) / "meta/info.json").is_file():
            raise FileNotFoundError(f"No LeRobot dataset at {self.dataset_root}")
        if self.dataset_root and (Path(self.dataset_root) / "SURGECOUNT_INCOMPLETE").exists():
            raise ValueError("Dataset preparation is incomplete")


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path is not None else {}
    known = {field.name for field in fields(TrainingConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = TrainingConfig(**values)
    config.validate()
    return config


def resolve_initialization(value):
    """Validate an explicit initialization choice without importing ML or Hub code.

    Ordinary strings are local directories. Only an explicit hf:// URI opts into
    a remote repository, so a missing relative path never becomes a Hub request.
    A checkpoint's policy type establishes adapter compatibility, not its origin.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "An explicit initialization checkpoint is required. Supply --base-model with a local "
            "checkpoint directory or an intentional hf:// Hub URI; initialization is not inferred."
        )
    if value.startswith("hf://"):
        repository = value.removeprefix("hf://")
        parts = repository.split("/")
        if len(parts) != 2 or any(
            not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}", part)
            or part.endswith((".", "-"))
            or "--" in part
            or ".." in part
            for part in parts
        ):
            raise ValueError("The hf:// initialization URI must contain a valid namespace and repository name")
        return "hub", repository
    directory = Path(value).expanduser().resolve()
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Local initialization checkpoint is missing {', '.join(missing)}: {directory}. "
            "Use an explicit hf:// URI only when a remote initialization is intended."
        )
    try:
        checkpoint_config = json.loads((directory / "config.json").read_text())
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid initialization checkpoint config.json: {directory}") from error
    checkpoint_type = checkpoint_config.get("type") if isinstance(checkpoint_config, dict) else None
    if checkpoint_type != "pi05":
        raise ValueError(f"Expected checkpoint policy type pi05, got {checkpoint_type!r}")
    return "local", str(directory)


def eligible_row_indices(episode_indices, task_indices, min_episode_index=251, max_episode_index=None):
    """Return dataset *positions*, not global `index` or episode metadata offsets."""
    import numpy as np

    episodes, tasks = np.asarray(episode_indices), np.asarray(task_indices)
    if episodes.ndim != 1 or tasks.ndim != 1 or len(episodes) != len(tasks):
        raise ValueError("Episode/task columns must be equally sized one-dimensional arrays")
    unknown = set(map(int, np.unique(tasks))) - set(TASK_LABELS)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    selected = (episodes >= min_episode_index) & ~np.isin(tasks, list(EXCLUDE_TASKS))
    if max_episode_index is not None:
        selected &= episodes <= max_episode_index
    indices = np.flatnonzero(selected).tolist()
    if not indices:
        raise ValueError("No eligible training frames after episode/task filtering")
    return indices


def mask_sample(sample):
    """Return a new sample with the inactive raw camera images exactly black."""
    task_id = int(sample["task_index"])
    if task_id not in TASK_LABELS:
        raise ValueError(f"Unknown task ID {task_id}")
    missing = set(ALL_CAMERAS) - set(sample)
    if missing:
        raise ValueError(f"Missing training cameras: {sorted(missing)}")
    if "task" in sample and sample["task"] != TASK_LABELS[task_id]:
        raise ValueError(f"Instruction mismatch for task {task_id}")
    result = dict(sample)
    active = TASK_CAMERA.get(task_id)
    for camera in ALL_CAMERAS:
        if camera != active:
            value = sample[camera]
            if hasattr(value, "new_zeros"):
                result[camera] = value.new_zeros(value.shape)
            else:
                import numpy as np

                result[camera] = np.zeros_like(value)
    return result


def masked_collate(batch, collate_fn=None):
    """Picklable standalone collator for callers constructing their own loader."""
    if collate_fn is None:
        from torch.utils.data import default_collate

        collate_fn = default_collate
    return collate_fn([mask_sample(sample) for sample in batch])


class TaskFilteredDataset:
    """Map selected anchors to native LeRobot rows, keeping temporal queries intact.

    This wrapper uses the already loaded Arrow table, including its exact subset
    ordering. Native LeRobot still resolves future action chunks and video times.
    It does not filter the HF table itself, which would break temporal indexing.
    """

    def __init__(self, dataset, min_episode_index=251, max_episode_index=None):
        import numpy as np

        self.dataset = dataset
        table = dataset.hf_dataset.data
        episodes = table.column("episode_index").to_numpy()
        tasks = table.column("task_index").to_numpy()
        self.indices = eligible_row_indices(episodes, tasks, min_episode_index, max_episode_index)
        self.num_frames = len(self.indices)
        self.num_episodes = len(np.unique(episodes[self.indices]))
        self.selection_report = {
            "loaded_frames": len(episodes),
            "eligible_anchor_frames": self.num_frames,
            "eligible_episode_count": self.num_episodes,
            "eligible_episode_indices": sorted(map(int, np.unique(episodes[self.indices]))),
            "eligible_anchors_per_task": {
                str(key): value for key, value in sorted(Counter(map(int, tasks[self.indices])).items())
            },
        }

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        return mask_sample(self.dataset[self.indices[index]])

    def __getattr__(self, name):
        # Never let PyTorch's batch-fetch optimization bypass __getitem__/masking.
        if name.startswith("__") or name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def validate_dataset_metadata(metadata):
    """Reject incompatible data before downloading a multi-gigabyte base model."""
    import numpy as np

    features, stats = metadata.features, metadata.stats or {}
    for name, width in (("action", 13), ("observation.state", 12)):
        if list(features.get(name, {}).get("shape", [])) != [width]:
            raise ValueError(
                f"Expected {name} shape [{width}]. Prepare a new action13 dataset with surgecount-data add-stop."
            )
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            values = np.asarray(stats.get(name, {}).get(key, []))
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError(f"Missing/invalid {width}-dimensional normalization statistic {name}/{key}")
    camera_keys = {key for key, value in features.items() if value.get("dtype") in ("image", "video")}
    if camera_keys != set(ALL_CAMERAS):
        raise ValueError(f"Expected exactly three canonical cameras, got {sorted(camera_keys)}")
    if float(stats["action"]["q99"][12]) <= float(stats["action"]["q01"][12]):
        raise ValueError("Stop-signal quantile range is degenerate; inspect dataset curation and normalization")


def validate_training_dataset(dataset, stop_last_n=10):
    """Check action/state dimensions, normalization, exact task strings and stop labels."""
    import numpy as np

    from surgecount.data import stop_targets

    validate_dataset_metadata(dataset.meta)
    if list(map(int, dataset.meta.tasks["task_index"])) != list(range(len(dataset.meta.tasks))):
        raise ValueError("Task metadata rows must be ordered by contiguous task_index")
    labels = {int(row["task_index"]): str(label) for label, row in dataset.meta.tasks.iterrows()}
    table = dataset.hf_dataset.data
    tasks = table.column("task_index").to_numpy()
    validate_task_labels(labels, np.unique(tasks))
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13 or not np.isfinite(actions).all():
        raise ValueError("All action rows must contain 13 finite values")
    expected = stop_targets(
        table.column("episode_index").to_numpy(),
        table.column("frame_index").to_numpy(),
        tasks,
        stop_last_n,
    )
    if not np.array_equal(actions[:, 12], expected):
        raise ValueError("Stop targets do not match the final frames of contiguous task segments")


def recipe_summary(config):
    return {
        "configuration": asdict(config),
        "full_finetune": True,
        "relative_actions": False,
        "action_dim": 13,
        "excluded_anchor_task_ids": sorted(EXCLUDE_TASKS),
        "camera_per_task": TASK_CAMERA,
        "episode_index_bounds": "inclusive, zero-based dataset metadata IDs",
        "historical_configuration_fully_recovered": False,
        "artifact_availability": ARTIFACT_AVAILABILITY,
        "release_assumptions": [
            "After 250 is implemented as episode_index >= 251; change min_episode_index if curation used ordinal numbering.",
            "10000 updates is the release training schedule; historical checkpoint artifacts are not bundled.",
            "Optimizer, scheduler, chunk size and worker defaults come from the vendored code, not a recovered historical run.",
            "Filtering excludes sampled anchors; native action chunks can extend across task boundaries.",
        ],
    }


@contextmanager
def training_hooks(train_module, release_config):
    """Scope integration to one training invocation, without patching torch globally."""
    original_dataset = train_module.make_dataset
    original_processors = train_module.make_pre_post_processors
    original_policy = train_module.make_policy

    def make_dataset(config):
        dataset = original_dataset(config)
        validate_training_dataset(dataset, release_config.stop_last_n_frames)
        filtered = TaskFilteredDataset(dataset, release_config.min_episode_index, release_config.max_episode_index)
        logging.info("SurgeCount selection: %s", json.dumps(filtered.selection_report))
        report = recipe_summary(release_config)
        report["selection"] = filtered.selection_report
        report["resolved_dataset_root"] = str(dataset.root)
        # TrainPipelineConfig.validate has already refused an existing output dir.
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "surgecount_run.json").write_text(json.dumps(report, indent=2) + "\n")
        return filtered

    def make_processors(policy_cfg, pretrained_path=None, **kwargs):
        # Base model weights are retained, but its processors may describe another
        # action width or relative actions. Fresh processors use action13 stats.
        return original_processors(
            policy_cfg=policy_cfg,
            pretrained_path=None,
            dataset_stats=kwargs["dataset_stats"],
        )

    def make_policy(**kwargs):
        policy = original_policy(**kwargs)
        frozen = [name for name, parameter in policy.named_parameters() if not parameter.requires_grad]
        if frozen:
            raise RuntimeError(f"Full finetuning required, but {len(frozen)} parameters are frozen: {frozen[:5]}")
        return policy

    with (
        patch.object(train_module, "make_dataset", make_dataset),
        patch.object(train_module, "make_pre_post_processors", make_processors),
        patch.object(train_module, "make_policy", make_policy),
    ):
        yield


def run_training(config):
    config.validate(require_local_data=True)
    initialization_kind, initialization_location = resolve_initialization(config.base_model)
    if Path(config.output_dir).exists():
        raise FileExistsError(f"Output already exists: {config.output_dir}; choose a new run directory")
    # Imports below may initialize frameworks; --dry-run never reaches here.
    from huggingface_hub import snapshot_download
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    import lerobot.scripts.lerobot_train as train_module

    metadata = LeRobotDatasetMetadata(
        config.dataset_repo_id, root=config.dataset_root, revision=config.dataset_revision
    )
    validate_dataset_metadata(metadata)
    episode_ids = list(map(int, metadata.episodes["episode_index"]))
    selected = [
        ep
        for ep in episode_ids
        if ep >= config.min_episode_index and (config.max_episode_index is None or ep <= config.max_episode_index)
    ]
    if not selected:
        raise ValueError("No episode metadata IDs match the configured inclusive bounds")
    base_path = initialization_location
    if initialization_kind == "hub":
        # A concrete snapshot path makes config and weights resolve to one revision.
        base_path = snapshot_download(
            initialization_location,
            revision=config.base_revision,
            allow_patterns=["config.json", "model.safetensors"],
        )
    policy = PreTrainedConfig.from_pretrained(base_path)
    if policy.type != "pi05":
        raise ValueError(f"Expected checkpoint policy type pi05, got {policy.type!r}")
    if policy.max_action_dim < 13 or policy.max_state_dim < 12:
        raise ValueError("Pretrained model padded dimensions are too small for the robot")
    # make_policy derives actual shapes/cameras from this dataset. The network's
    # padded 32-dim projection retains its pretrained tensor shapes.
    policy.input_features, policy.output_features = {}, {}
    policy.pretrained_path = Path(base_path)
    policy.dtype, policy.device = config.dtype, config.device
    policy.gradient_checkpointing = config.gradient_checkpointing
    policy.freeze_vision_encoder = False
    policy.train_expert_only = False
    policy.use_peft = False
    policy.use_relative_actions = False
    policy.push_to_hub = False
    policy.chunk_size = config.chunk_size
    policy.n_action_steps = config.chunk_size
    for field in (
        "optimizer_lr",
        "optimizer_weight_decay",
        "scheduler_warmup_steps",
        "scheduler_decay_steps",
        "scheduler_decay_lr",
    ):
        setattr(policy, field, getattr(config, field))
    if hasattr(policy, "drop_n_last_frames"):
        raise RuntimeError(
            "This integration expects the vendored PI05 random-anchor loader; incompatible sampler config"
        )
    pipeline = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=config.dataset_repo_id,
            root=config.dataset_root,
            revision=config.dataset_revision,
            episodes=selected,
            video_backend=config.video_backend,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=Path(config.output_dir),
        job_name="skill_singlecam",
        seed=config.seed,
        batch_size=config.batch_size,
        steps=config.steps,
        save_freq=config.save_freq,
        num_workers=config.num_workers,
        eval_freq=0,
        log_freq=20,
        wandb=WandBConfig(enable=False),
    )
    with training_hooks(train_module, config):
        # Bypass the external CLI parser while retaining upstream validation,
        # optimizer/accelerator handling, checkpoints and saved processors.
        train_module.train.__wrapped__(pipeline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON release configuration; omitted uses documented defaults")
    parser.add_argument("--dataset-root", help="Concrete prepared local LeRobot action13 tree")
    parser.add_argument("--base-model", help="Explicit local initialization checkpoint or intentional hf:// Hub URI")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-episode-index", type=int)
    parser.add_argument("--max-episode-index", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print recipe and optional local counts; never contact Hub"
    )
    args = parser.parse_args(argv)
    values = vars(args).copy()
    path, dry_run = values.pop("config"), values.pop("dry_run")
    config = load_config(path, **values)
    report = recipe_summary(config)
    if dry_run and config.dataset_root:
        from surgecount.data import inspect_dataset

        report["local_dataset_audit"] = inspect_dataset(
            config.dataset_root, config.min_episode_index, config.max_episode_index
        )
    print(json.dumps(report, indent=2))
    if not dry_run:
        run_training(config)


if __name__ == "__main__":
    main()
