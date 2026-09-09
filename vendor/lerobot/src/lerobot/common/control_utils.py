# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def _select_keyboard_backend():
    """Return a `keyboard` namespace exposing `.Listener` and `.Key`.

    pynput's Linux backend is X11-only and silently captures zero events on
    Wayland sessions. Fall back to an evdev-backed shim there.
    """
    import os

    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        try:
            from lerobot.utils.evdev_keyboard import keyboard as evdev_keyboard

            logging.info("Wayland session detected; using evdev keyboard listener.")
            return evdev_keyboard
        except Exception as e:
            logging.warning(
                "evdev keyboard backend unavailable (%s); falling back to pynput.", e
            )
    from pynput import keyboard

    return keyboard


def init_keyboard_listener(
    subtasks: dict[str, str] | None = None,
    subtask_voices: dict[str, str] | None = None,
    play_sounds: bool = False,
):
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    Two modes:

    - Default (``subtasks=None``): keys ``1``/``2``/``3`` end the current episode /
      stop recording / re-record. ``ESC`` also stops. Arrow keys kept as fallback.
    - Subtask-labeling (``subtasks`` is a ``{key_char: instruction_string}`` dict):
      keys ``1``–``9`` (whichever are present in ``subtasks``) set
      ``events["current_task"]`` to the matching instruction string. If the
      instruction contains the literal ``{N}`` placeholder, the key press enters
      a pending state instead, and the next ``a``/``b``/``c`` press resolves
      ``{N}`` to slot ``1``/``2``/``3``. The record
      loop reads this to tag each frame's ``task`` field. Episode control moves
      off the number keys to ``Enter`` (end) / ``ESC`` (stop) / ``Backspace``
      (re-record). Arrow keys remain as fallback for end/re-record.

    Returns:
        Tuple of (pynput listener or ``None`` in headless env, events dict).
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["current_task"] = None
    events["operator_revision"] = 0
    from lerobot.common.operator_events import operator_activity, operator_lock

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    keyboard = _select_keyboard_backend()

    if subtasks:
        from lerobot.utils.utils import log_say

        # Runtime slot selection: a/b/c -> 1/2/3 for any {N}-templated subtask.
        SLOT_KEYS = {"a": 1, "b": 2, "c": 3}
        pending = {"template": None, "key": None}

        def _voice(key_char: str, fallback: str) -> str:
            if subtask_voices and key_char in subtask_voices:
                return subtask_voices[key_char]
            return fallback

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char in subtasks:
                    task_str = subtasks[key.char]
                    if "{N}" in task_str:
                        events["operator_slot_pending"] = True
                        pending["template"] = task_str
                        pending["key"] = key.char
                        print(
                            f"'{key.char}' pressed. Now press 'a'/'b'/'c' to choose slot 1/2/3."
                        )
                        log_say(_voice(key.char, f"key {key.char}"), play_sounds)
                    else:
                        events["operator_slot_pending"] = False
                        pending["template"] = None
                        pending["key"] = None
                        print(f"'{key.char}' pressed. Subtask -> {task_str}")
                        events["current_task"] = task_str
                        events.pop("pending_planner_decision", None)
                        log_say(_voice(key.char, f"key {key.char}"), play_sounds)
                elif pending["template"] is not None and hasattr(key, 'char') and key.char in SLOT_KEYS:
                    slot = SLOT_KEYS[key.char]
                    resolved = pending["template"].replace("{N}", str(slot))
                    print(
                        f"'{key.char}' pressed (after '{pending['key']}'). Subtask -> {resolved}"
                    )
                    events["current_task"] = resolved
                    events["operator_slot_pending"] = False
                    events.pop("pending_planner_decision", None)
                    pending["template"] = None
                    pending["key"] = None
                    log_say(f"slot {slot}", play_sounds)
                elif key == keyboard.Key.space:
                    # Confirm a pending Gemini decision: promote it into
                    # events["current_task"]. No-op if nothing pending.
                    pdec = events.get("pending_planner_decision")
                    if pdec is not None and pdec.get("label"):
                        events["current_task"] = pdec["label"]
                        events["operator_slot_pending"] = False
                        events.pop("pending_planner_decision", None)
                        pending["template"] = None
                        pending["key"] = None
                        print(f"SPACE pressed. Confirmed planner pick -> {pdec.get('chosen_key')} ({pdec['label']!r})")
                        log_say("confirmed", play_sounds)
                    else:
                        print("SPACE pressed but no pending planner decision.")
                elif hasattr(key, 'char') and key.char == 'e':
                    pending["template"] = None
                    pending["key"] = None
                    events["current_task"] = "subtask finished"
                    events["operator_slot_pending"] = False
                    events.pop("pending_planner_decision", None)
                    print("'e' pressed. Subtask -> subtask finished")
                    log_say("subtask finished", play_sounds)
                elif hasattr(key, 'char') and key.char == 'h':
                    pending["template"] = None
                    pending["key"] = None
                    events["current_task"] = None
                    events["operator_slot_pending"] = False
                    events.pop("pending_planner_decision", None)
                    events["go_home"] = True
                    events["exit_early"] = True
                    print("'h' pressed. Going home and restarting episode...")
                    log_say("going home", play_sounds)
                elif hasattr(key, 'char') and key.char in (']', '+'):
                    # Mark the just-completed subtask as a SUCCESS. Consumed by
                    # the high-level planner (if any) to inform its next decision
                    # and written to the planner's JSONL trace as feedback.
                    events["last_subtask_outcome"] = {
                        "outcome": "success",
                        "note": None,
                    }
                    print(f"'{key.char}' pressed. Marked last subtask as SUCCESS.")
                    log_say("success", play_sounds)
                elif hasattr(key, 'char') and key.char in ('[', '-'):
                    # Mark the just-completed subtask as a FAIL.
                    events["last_subtask_outcome"] = {
                        "outcome": "fail",
                        "note": None,
                    }
                    print(f"'{key.char}' pressed. Marked last subtask as FAIL.")
                    log_say("fail", play_sounds)
                elif key == keyboard.Key.enter:
                    print("Enter pressed. Ending episode...", flush=True)
                    try:
                        with open("/tmp/lerobot_keylog", "a") as _kf:
                            import time as _t
                            _kf.write(f"{_t.strftime('%H:%M:%S')} ENTER -> exit_early=True\n")
                    except Exception:
                        pass
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("ESC pressed. Stopping recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.backspace:
                    print("Backspace pressed. Re-recording episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.right:
                    print("Right arrow pressed. Ending episode...")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("Left arrow pressed. Re-recording episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")
    else:
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == '1':
                    print("'1' pressed. Ending episode...")
                    events["exit_early"] = True
                elif hasattr(key, 'char') and key.char == '2':
                    print("'2' pressed. Stopping recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                elif hasattr(key, 'char') and key.char == '3':
                    print("'3' pressed. Re-recording episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("ESC pressed. Stopping recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.right:
                    print("Right arrow pressed. Ending episode...")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("Left arrow pressed. Re-recording episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")

    raw_on_press = on_press

    def on_press(key):
        # Hold the same lock used by planner-result installation. A revision
        # also catches inputs that set a task and subsequently clear it.
        with operator_lock(events):
            operator_activity(events, preserve_pending=(key == keyboard.Key.space))
            raw_on_press(key)
            if events.get("exit_early") or events.get("stop_recording"):
                events.pop("pending_planner_decision", None)
                events["operator_slot_pending"] = False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Stdin-reader fallback: if the evdev listener silently dies or doesn't
    # see your keyboard, you can still end an episode by pressing Enter in
    # the script's own terminal. Each blank line typed sets exit_early=True.
    # Type `q<Enter>` to fully stop, `r<Enter>` to re-record.
    def _stdin_reader():
        import sys, time as _t
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                try:
                    with open("/tmp/lerobot_keylog", "a") as _kf:
                        _kf.write(f"{_t.strftime('%H:%M:%S')} [stdin] line={cmd!r}\n")
                except Exception:
                    pass
                with operator_lock(events):
                    operator_activity(events)
                    events["operator_slot_pending"] = False
                    if cmd == "q":
                        print("[stdin] q -> Stopping recording...", flush=True)
                        events["stop_recording"] = True
                        events["exit_early"] = True
                    elif cmd == "r":
                        print("[stdin] r -> Re-recording episode...", flush=True)
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                    else:
                        print("[stdin] Enter -> Ending episode...", flush=True)
                        events["exit_early"] = True
        except Exception:
            pass

    import threading as _th
    _stdin_thread = _th.Thread(target=_stdin_reader, name="stdin-fallback", daemon=True)
    _stdin_thread.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
