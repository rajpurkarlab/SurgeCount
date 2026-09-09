"""Exercise production queue and planner code without torch, models, or devices."""

import ast
from collections import deque
from contextlib import redirect_stdout
import importlib.util
import io
import logging
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

VENDOR = Path(__file__).resolve().parents[1] / "vendor/lerobot/src/lerobot"


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, VENDOR / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


EVENTS = load_module("common/operator_events.py", "surgecount_test_operator_events")


def production_function(relative, name, class_name=None, namespace=None):
    """Compile the exact production body; omit heavy module imports/decorators."""
    path = VENDOR / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class FakeChunk:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, slices):
        return FakeChunk(self.values[slices[1]])

    def transpose(self, first, second):
        return self.values


class DynamicQueueTests(unittest.TestCase):
    def setUp(self):
        self.select = production_function(
            "policies/pi05/modeling_pi05.py", "select_action", "PI05Policy", {"deque": deque}
        )

    def policy(self, size):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(n_action_steps=size),
            _action_queue=deque(maxlen=5),
            _rtc_enabled=lambda: False,
            eval=lambda: None,
            predict_action_chunk=lambda batch: FakeChunk(range(50)),
        )

    def test_five_to_thirty_retains_first_action_and_whole_chunk(self):
        policy = self.policy(30)
        self.assertEqual([self.select(policy, {}) for _ in range(30)], list(range(30)))
        self.assertEqual(policy._action_queue.maxlen, 30)

    def test_five_to_forty_retains_first_action_and_whole_chunk(self):
        policy = self.policy(40)
        self.assertEqual([self.select(policy, {}) for _ in range(40)], list(range(40)))
        self.assertEqual(policy._action_queue.maxlen, 40)

    def test_reconfiguration_preserves_pending_actions_until_queue_empty(self):
        policy = self.policy(30)
        policy._action_queue.extend([98, 99])
        self.assertEqual(self.select(policy, {}), 98)
        self.assertEqual(policy._action_queue.maxlen, 5)
        self.assertEqual(self.select(policy, {}), 99)
        self.assertEqual(self.select(policy, {}), 0)
        self.assertEqual(policy._action_queue.maxlen, 30)


class PlannerPriorityTests(unittest.TestCase):
    def setUp(self):
        self.events = {"current_task": None, "operator_revision": 0}
        self.calls = 0
        self.mutation = lambda events: None
        namespace = {
            "time": time,
            "LOGGER": logging.getLogger(__name__),
            "_slots_to_list": lambda value: sorted(value or []),
            "SKILL_VOICE": {},
            "log_say": lambda *args, **kwargs: None,
            "PlannerDecision": lambda **values: types.SimpleNamespace(**values),
            "planner_snapshot": EVENTS.planner_snapshot,
            "planner_result_current": EVENTS.planner_result_current,
            "apply_planner_result": EVENTS.apply_planner_result,
        }
        self.maybe_call = production_function("common/gemini_planner.py", "maybe_call", "GeminiPlanner", namespace)

        def call_once(*args, **kwargs):
            self.calls += 1
            self.mutation(self.events)
            return types.SimpleNamespace(
                chosen_key="1", rationale="test", visible_blue_confidence=1.0, latency_s=0, label=None
            )

        self.planner = types.SimpleNamespace(
            confirm_required=False,
            _last_chosen_key=None,
            _labeled_history=deque(),
            _last_call_t=0,
            cooldown_s=0,
            _call_once=call_once,
            _resolve_label=lambda key: "planner skill",
            _wrong_grab_count=0,
            _log_decision=lambda *args: None,
            dry_run=False,
            speak=False,
        )
        self.obs = {"left_top": object(), "left_left_front": object(), "right_right_front": object()}

    def run_planner(self):
        return self.maybe_call(self.planner, self.events, self.obs, "gauze_placement")

    def test_unchanged_intent_applies_planner_result(self):
        self.run_planner()
        self.assertEqual(self.events["current_task"], "planner skill")

    def test_pause_during_api_call_is_not_overwritten(self):
        def pause(events):
            EVENTS.operator_activity(events)
            events["current_task"] = "subtask finished"

        self.mutation = pause
        self.assertIsNone(self.run_planner())
        self.assertEqual(self.events["current_task"], "subtask finished")

    def test_operator_skill_without_revision_still_wins(self):
        self.mutation = lambda events: events.update(current_task="operator skill")
        self.assertIsNone(self.run_planner())
        self.assertEqual(self.events["current_task"], "operator skill")

    def test_changed_then_cleared_intent_invalidates_old_response(self):
        def switch_and_clear(events):
            EVENTS.operator_activity(events)
            events["current_task"] = "operator skill"
            EVENTS.operator_activity(events)
            events["current_task"] = None

        self.mutation = switch_and_clear
        self.assertIsNone(self.run_planner())
        self.assertIsNone(self.events["current_task"])
        self.assertIsNone(self.planner._last_chosen_key)

    def test_exit_during_api_does_not_leave_confirmable_decision(self):
        self.planner.confirm_required = True
        self.mutation = lambda events: events.update(exit_early=True, stop_recording=True)
        self.assertIsNone(self.run_planner())
        self.assertNotIn("pending_planner_decision", self.events)

    def test_no_api_calls_while_exiting_paused_or_selecting_slot(self):
        for flag in ("exit_early", "stop_recording", "rerecord_episode", "go_home", "operator_slot_pending"):
            with self.subTest(flag=flag):
                self.events = {"current_task": None, flag: True}
                self.assertIsNone(self.run_planner())
        self.events = {"current_task": "subtask finished"}
        self.assertIsNone(self.run_planner())
        self.events = {"current_task": None, "pending_planner_decision": {"label": "old suggestion"}}
        self.assertIsNone(self.run_planner())
        self.assertEqual(self.calls, 0)

    def test_operator_change_immediately_before_commit_wins(self):
        # Mutation in logging happens after the first stale-response check but
        # before final installation, exercising the atomic commit guard.
        def log_decision(*args):
            EVENTS.operator_activity(self.events)
            self.events["current_task"] = "operator skill"

        self.planner._log_decision = log_decision
        self.assertIsNone(self.run_planner())
        self.assertEqual(self.events["current_task"], "operator skill")


class KeyboardIntegrationTests(unittest.TestCase):
    def test_real_callback_revisions_and_stale_pending_clear_on_pause_escape(self):
        key = types.SimpleNamespace(
            space=object(), enter=object(), esc=object(), backspace=object(), right=object(), left=object()
        )

        class Listener:
            def __init__(self, on_press):
                self.on_press = on_press

            def start(self):
                pass

        backend = types.SimpleNamespace(Key=key, Listener=Listener)
        init = production_function(
            "common/control_utils.py",
            "init_keyboard_listener",
            namespace={"is_headless": lambda: False, "_select_keyboard_backend": lambda: backend, "logging": logging},
        )
        utils = types.ModuleType("lerobot.utils.utils")
        utils.log_say = lambda *args, **kwargs: None
        modules = {"lerobot.common.operator_events": EVENTS, "lerobot.utils.utils": utils}
        with patch.dict(sys.modules, modules), patch("threading.Thread.start"), redirect_stdout(io.StringIO()):
            listener, events = init({"1": "skill one", "2": "slot {N}"})
            events["pending_planner_decision"] = {"label": "old suggestion"}
            listener.on_press(types.SimpleNamespace(char="e"))
            self.assertEqual(events["operator_revision"], 1)
            self.assertEqual(events["current_task"], "subtask finished")
            self.assertNotIn("pending_planner_decision", events)
            listener.on_press(types.SimpleNamespace(char="2"))
            self.assertTrue(events["operator_slot_pending"])
            listener.on_press(types.SimpleNamespace(char="b"))
            self.assertFalse(events["operator_slot_pending"])
            self.assertEqual(events["current_task"], "slot 2")
            events["pending_planner_decision"] = {"label": "stale after exit"}
            listener.on_press(key.esc)
            self.assertTrue(events["stop_recording"])
            self.assertNotIn("pending_planner_decision", events)
            self.assertEqual(events["operator_revision"], 4)


if __name__ == "__main__":
    unittest.main()
