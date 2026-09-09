"""Offline configuration and launch construction for the SurgeCount robot.

This module deliberately imports no robot, camera, torch, or cloud SDK code.
Checks may read local JSON and checkpoint metadata, but never open devices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

JOINTS = tuple(
    f"{arm}_{joint}.pos"
    for arm in ("left", "right")
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
)
IMAGE_RENAME = {
    "observation.images.left_left_front": "observation.images.left_wrist_0_rgb",
    "observation.images.left_top": "observation.images.base_0_rgb",
    "observation.images.right_right_front": "observation.images.right_wrist_0_rgb",
}
SKILLS = {"1", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8", "9"}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open() as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _keys(value: dict, allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def _positive(value: Any, name: str, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_hardware(hw: dict, recording: bool = False, assets: bool = False) -> None:
    _keys(hw, {"robot", "teleop", "home_pose_file", "home_duration_s"}, "hardware")
    if "robot" not in hw:
        raise ValueError("hardware.robot is required")
    devices = [(hw["robot"], "robot", "bi_so_follower")]
    if recording:
        if "teleop" not in hw:
            raise ValueError("Recording requires hardware.teleop with two leader arms")
        devices.append((hw["teleop"], "teleop", "bi_so_leader"))
    ports = []
    for device, name, device_type in devices:
        _keys(device, {"type", "id", "calibration_dir", "left_arm_config", "right_arm_config"}, name)
        if device.get("type") != device_type:
            raise ValueError(f"{name}.type must be {device_type}")
        for key in ("id", "calibration_dir"):
            _text(device.get(key), f"{name}.{key}")
        for side in ("left", "right"):
            arm = device.get(f"{side}_arm_config", {})
            _keys(
                arm,
                {"port", "cameras", "use_degrees", "max_relative_target", "disable_torque_on_disconnect"},
                f"{name}.{side}",
            )
            _text(arm.get("port"), f"{name}.{side}.port")
            ports.append(arm["port"])
            if not isinstance(arm.get("use_degrees"), bool):
                raise ValueError(f"{name}.{side}.use_degrees must explicitly match the collected data")
            if "max_relative_target" in arm and arm["max_relative_target"] is not None:
                _positive(arm["max_relative_target"], "max_relative_target")
            if name == "robot":
                expected = {"left_front", "top"} if side == "left" else {"right_front"}
                cameras = arm.get("cameras", {})
                if set(cameras) != expected:
                    raise ValueError(f"{name}.{side}.cameras must be {sorted(expected)}")
                for camera in cameras.values():
                    _keys(camera, {"type", "index_or_path", "width", "height", "fps", "warmup_s", "fourcc"}, "camera")
                    if camera.get("type") != "opencv":
                        raise ValueError("The release configuration supports OpenCV cameras")
                    if not isinstance(camera.get("index_or_path"), (str, int)) or isinstance(
                        camera["index_or_path"], bool
                    ):
                        raise ValueError("Camera index_or_path must be a device path or numeric index")
                    for field in ("width", "height", "fps"):
                        _positive(camera.get(field), f"camera.{field}", integer=True)
                    if (camera["width"], camera["height"]) != (640, 480):
                        raise ValueError("Final590 cameras require width=640, height=480")
            if assets:
                calibration = Path(device["calibration_dir"]).expanduser() / f"{device['id']}_{side}.json"
                values = load_json(calibration)
                expected_joints = {joint.split("_", 1)[1][:-4] for joint in JOINTS if joint.startswith(f"{side}_")}
                if set(values) != expected_joints:
                    raise ValueError(f"Calibration must contain the six {side} motors: {calibration}")
                for motor in values.values():
                    if not {"id", "drive_mode", "homing_offset", "range_min", "range_max"} <= set(motor):
                        raise ValueError(f"Incomplete motor calibration: {calibration}")
    if len(ports) != len(set(ports)):
        raise ValueError("Each follower/leader arm needs a distinct serial port")
    _positive(hw.get("home_duration_s", 2.0), "home_duration_s")
    if hw.get("home_pose_file") and assets:
        read_home_pose(hw["home_pose_file"])


def read_home_pose(path: str | Path) -> dict[str, float]:
    pose = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in pose:
            raise ValueError(f"Duplicate home joint: {key}")
        pose[key.strip()] = float(value)
    if set(pose) != set(JOINTS) or not all(math.isfinite(v) for v in pose.values()):
        raise ValueError("Home pose must contain exactly the twelve finite joint/gripper targets")
    return pose


def validate_config(cfg: dict, deployment: bool, hw: dict, assets: bool = False) -> None:
    _keys(cfg, {"dataset", "resume", "display_data", "play_sounds", "deployment"}, "session")
    data = cfg.get("dataset", {})
    _keys(
        data,
        {
            "repo_id",
            "root",
            "fps",
            "num_episodes",
            "episode_time_s",
            "reset_time_s",
            "reset_wait_enter",
            "streaming_encoding",
            "encoder_threads",
            "vcodec",
            "single_task",
        },
        "dataset",
    )
    for key in ("repo_id", "root"):
        _text(data.get(key), f"dataset.{key}")
    if len(data["repo_id"].split("/")) != 2:
        raise ValueError("dataset.repo_id must have namespace/name form")
    if data.get("fps") != 10:
        raise ValueError("Final590 recording and deployment use dataset.fps=10")
    for key in ("num_episodes", "episode_time_s", "reset_time_s"):
        _positive(data.get(key), f"dataset.{key}", integer=key == "num_episodes")
    if not deployment:
        if cfg.get("deployment"):
            raise ValueError("Recording cannot include a deployment policy")
        return
    dep = cfg.get("deployment", {})
    _keys(
        dep,
        {
            "checkpoint",
            "device",
            "mode",
            "camera_mask",
            "n_action_steps",
            "num_inference_steps",
            "interpolation_multiplier",
            "stop_mode",
            "stop_window",
            "dynamic_chunk",
            "timeouts",
            "planner",
            "image_keys",
        },
        "deployment",
    )
    _text(dep.get("checkpoint"), "deployment.checkpoint")
    if dep.get("mode") not in {"manual", "gemini"}:
        raise ValueError("deployment.mode must be manual or gemini")
    if dep.get("device") not in {"cpu", "cuda", "mps"}:
        raise ValueError("deployment.device must be cpu, cuda, or mps")
    if dep.get("image_keys") not in {"native", "openpi"}:
        raise ValueError("deployment.image_keys must match checkpoint keys: native or openpi")
    if dep.get("camera_mask") is not True:
        raise ValueError("The single-camera checkpoint requires camera_mask=true")
    for key in ("n_action_steps", "num_inference_steps", "interpolation_multiplier", "stop_window"):
        _positive(dep.get(key), key, integer=True)
    if dep["stop_mode"] not in {"lookahead", "consecutive"}:
        raise ValueError("stop_mode must be lookahead or consecutive")
    if dep["n_action_steps"] > 50 or dep["stop_window"] > dep["n_action_steps"]:
        raise ValueError("stop_window <= n_action_steps <= training chunk size 50 is required")
    if dep["stop_mode"] == "lookahead" and dep["interpolation_multiplier"] < 2:
        raise ValueError("Lookahead requires interpolation_multiplier >= 2 in this runtime")
    dyn = dep.get("dynamic_chunk", {})
    _keys(dyn, {"enabled", "minimum", "maximum", "step", "calls_per_step", "keys"}, "dynamic_chunk")
    for key in ("minimum", "maximum", "step", "calls_per_step"):
        _positive(dyn.get(key), f"dynamic_chunk.{key}", integer=True)
    if not 1 <= dyn["minimum"] <= dyn["maximum"] <= 50:
        raise ValueError("dynamic_chunk requires 1 <= minimum <= maximum <= 50")
    if not set(dyn.get("keys", [])) <= SKILLS:
        raise ValueError("dynamic_chunk keys must be skill keyboard identifiers")
    if dyn.get("enabled") and (dep["interpolation_multiplier"] < 2 or dyn["minimum"] < dep["stop_window"]):
        raise ValueError("Dynamic chunking requires interpolation and minimum >= stop_window")
    timeouts = dep.get("timeouts", {})
    if not set(timeouts) <= SKILLS:
        raise ValueError("Timeout keys must be skill keyboard identifiers")
    for key, value in timeouts.items():
        _positive(value, f"timeout.{key}")
    if (timeouts or dep["mode"] == "gemini") and not hw.get("home_pose_file"):
        raise ValueError("Recovery requires an explicit machine-specific home_pose_file")
    planner = dep.get("planner", {})
    _keys(planner, {"model", "cooldown_s", "confirm_required", "prompt_style", "thinking_level"}, "planner")
    _text(planner.get("model"), "planner.model")
    _positive(planner.get("cooldown_s"), "planner.cooldown_s")
    if planner.get("prompt_style") not in {"detailed", "minimal"}:
        raise ValueError("planner.prompt_style must be detailed or minimal")
    if assets:
        checkpoint = Path(dep["checkpoint"]).expanduser()
        policy = load_json(checkpoint / "config.json")
        if policy.get("type") != "pi05" or policy.get("output_features", {}).get("action", {}).get("shape") != [13]:
            raise ValueError("Checkpoint must be pi05 with action.shape=[13]")
        expected = set(IMAGE_RENAME.values()) if dep["image_keys"] == "openpi" else set(IMAGE_RENAME)
        actual = {k for k in policy.get("input_features", {}) if k.startswith("observation.images.")}
        if actual != expected:
            raise ValueError(f"Checkpoint camera keys mismatch: expected {sorted(expected)}, got {sorted(actual)}")
        for filename in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (checkpoint / filename).is_file():
                raise ValueError(f"Missing checkpoint artifact: {checkpoint / filename}")
        if dep["n_action_steps"] > policy.get("chunk_size", 50):
            raise ValueError("n_action_steps exceeds checkpoint chunk_size")


def _flag(key: str, value: Any) -> str:
    return f"--{key}={json.dumps(value, separators=(',', ':')) if not isinstance(value, str) else value}"


def build_launch(cfg: dict, hw: dict, deployment: bool) -> tuple[list[str], dict[str, str]]:
    """Construct an argv vector, never a shell command; API secrets are not included."""
    validate_hardware(hw, recording=not deployment)
    validate_config(cfg, deployment, hw)
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_record"]
    for name in ("robot",) if deployment else ("robot", "teleop"):
        for key, value in hw[name].items():
            if key.endswith("_arm_config"):
                for subkey, subvalue in value.items():
                    command.append(_flag(f"{name}.{key}.{subkey}", subvalue))
            else:
                command.append(_flag(f"{name}.{key}", value))
    for key, value in cfg["dataset"].items():
        command.append(_flag(f"dataset.{key}", value))
    command.extend(["--dataset.push_to_hub=false", "--enable_subtasks=true", "--calibrate_on_connect=false"])
    from surgecount.tasks import TASK_LABELS

    command.append(_flag("canonical_task_labels", [TASK_LABELS[key] for key in sorted(TASK_LABELS)]))
    if "single_task" not in cfg["dataset"]:
        command.append("--dataset.single_task=gauze_placement")
    for key in ("resume", "display_data", "play_sounds"):
        command.append(_flag(key, cfg.get(key, False)))
    if hw.get("home_pose_file"):
        command.append(_flag("home_pose_file", hw["home_pose_file"]))
        command.append(_flag("home_duration_s", hw.get("home_duration_s", 2.0)))
    env = {"DYN_CHUNK": "0", "DYN_TIMEOUT": "0", "LA_GRIPPER_OPEN_OFFSET": "0", "DYN_CHUNK_START_MIN": "0"}
    if deployment:
        dep = cfg["deployment"]
        for key in ("checkpoint", "device", "n_action_steps", "num_inference_steps"):
            command.append(_flag(f"policy.{'path' if key == 'checkpoint' else key}", dep[key]))
        for key, value in {
            "interpolation_multiplier": dep["interpolation_multiplier"],
            "cam_mask_per_subtask": True,
            "stop_signal_dim": 12,
            "stop_signal_threshold": dep["stop_window"],
            "stop_signal_mode": dep["stop_mode"],
        }.items():
            command.append(_flag(key, value))
        if dep["image_keys"] == "openpi":
            command.append(_flag("dataset.rename_map", IMAGE_RENAME))
        dyn = dep["dynamic_chunk"]
        env.update(
            {
                "DYN_CHUNK": "1" if dyn["enabled"] else "0",
                "DYN_CHUNK_KEYS": ",".join(dyn["keys"]),
                "DYN_CHUNK_MIN": str(dyn["minimum"]),
                "DYN_CHUNK_MAX": str(dyn["maximum"]),
                "DYN_CHUNK_STEP": str(dyn["step"]),
                "DYN_CHUNK_CALLS_PER_STEP": str(dyn["calls_per_step"]),
            }
        )
        timeouts = dep["timeouts"]
        env.update(
            {
                "DYN_TIMEOUT": "1" if timeouts else "0",
                "DYN_TIMEOUT_KEYS": ",".join(timeouts),
                "DYN_TIMEOUT_HOME_DURATION_S": str(hw.get("home_duration_s", 2.0)),
            }
        )
        env.update({f"DYN_TIMEOUT_S_{key}": str(value) for key, value in timeouts.items()})
        if dep["mode"] == "gemini":
            plan = dep["planner"]
            command.extend(
                [
                    "--planner=gemini",
                    _flag("planner_model", plan["model"]),
                    _flag("planner_cooldown_s", plan["cooldown_s"]),
                    _flag("planner_confirm_required", plan["confirm_required"]),
                ]
            )
            env.update({"GEMINI_PROMPT_STYLE": plan["prompt_style"], "GEMINI_THINKING_LEVEL": plan["thinking_level"]})
    return command, env


def cli(deployment: bool, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SurgeCount deployment" if deployment else "SurgeCount teleoperation recording"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate schema and print launch; no imports, hardware, downloads, or API calls",
    )
    parser.add_argument(
        "--check", action="store_true", help="Also validate local calibration and checkpoint metadata; never connect"
    )
    if deployment:
        parser.add_argument("--checkpoint", type=Path, help="Local checkpoint override")
        parser.add_argument("--mode", choices=("manual", "gemini"))
    args = parser.parse_args(argv)
    try:
        cfg, hw = load_json(args.config), load_json(args.hardware)
        if deployment and args.checkpoint:
            cfg["deployment"]["checkpoint"] = str(args.checkpoint.expanduser().resolve())
        if deployment and args.mode:
            cfg["deployment"]["mode"] = args.mode
        validate_hardware(hw, recording=not deployment, assets=args.check or not args.dry_run)
        validate_config(cfg, deployment, hw, assets=args.check or not args.dry_run)
        command, env = build_launch(cfg, hw, deployment)
        if args.dry_run or args.check:
            print(
                json.dumps(
                    {"validated": True, "hardware_connected": False, "argv": command, "runtime_environment": env},
                    indent=2,
                )
            )
            return 0
        if deployment and cfg["deployment"]["mode"] == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Export GEMINI_API_KEY before Gemini deployment; keys are never read from files")
        # Reject untracked ambient overrides from experimental workstation launchers.
        process_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(
                ("DYN_", "STOP_SIGNAL_THRESHOLD_", "STOP_EARLY_", "LA_GRIPPER_", "BOUNDARY_", "LONG_PAUSE_")
            )
        }
        process_env.update(env)
        if deployment:
            # Deployment is local-only; provision the checkpoint and tokenizer cache beforehand.
            process_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        return subprocess.run(command, env=process_env, check=False).returncode
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2
