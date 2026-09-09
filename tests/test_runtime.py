"""No hardware, model downloads, or planner calls are made by these tests."""

import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from surgecount.runtime import build_launch, cli, load_json, read_home_pose, validate_config, validate_hardware, JOINTS

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/lerobot/src/lerobot"


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_json(ROOT / "configs/deploy.json")
        self.hw = load_json(ROOT / "configs/robot.example.json")

    def test_dry_run_never_launches_hardware_or_planner(self):
        with patch("surgecount.runtime.subprocess.run") as run, redirect_stdout(io.StringIO()) as output:
            result = cli(
                True,
                [
                    "--config",
                    str(ROOT / "configs/deploy.json"),
                    "--hardware",
                    str(ROOT / "configs/robot.example.json"),
                    "--mode",
                    "gemini",
                    "--dry-run",
                ],
            )
        self.assertEqual(result, 0)
        run.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())["hardware_connected"])

    def test_invalid_asset_check_never_launches(self):
        with (
            patch("surgecount.runtime.subprocess.run") as run,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            cli(
                True,
                [
                    "--config",
                    str(ROOT / "configs/deploy.json"),
                    "--hardware",
                    str(ROOT / "configs/robot.example.json"),
                    "--check",
                ],
            )
        run.assert_not_called()

    def test_launch_preserves_stop_dimension_and_external_config(self):
        argv, env = build_launch(self.cfg, self.hw, True)
        self.assertIn("--stop_signal_dim=12", argv)
        self.assertIn("--calibrate_on_connect=false", argv)
        self.assertIn("--dataset.push_to_hub=false", argv)
        self.assertEqual((env["DYN_CHUNK_MAX"], env["DYN_CHUNK_MIN"], env["DYN_CHUNK_STEP"]), ("30", "5", "5"))
        self.assertFalse(any("--teleop" in arg for arg in argv))

    def test_example_uses_local_anonymous_artifact_placeholders(self):
        self.assertEqual(self.cfg["dataset"]["repo_id"], "local/evaluation")
        self.assertEqual(self.cfg["dataset"]["root"], "outputs/evaluation")
        self.assertEqual(self.cfg["deployment"]["checkpoint"], "/absolute/path/to/checkpoint")

    def test_shell_metacharacters_are_literal_argument_content(self):
        self.cfg["deployment"]["checkpoint"] = "/tmp/checkpoint $(touch BAD);echo X"
        argv, _ = build_launch(self.cfg, self.hw, True)
        self.assertIn("--policy.path=/tmp/checkpoint $(touch BAD);echo X", argv)

    def test_camera_map_and_duplicate_ports_rejected(self):
        self.hw["robot"]["right_arm_config"]["port"] = self.hw["robot"]["left_arm_config"]["port"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_hardware(self.hw)
        self.hw = load_json(ROOT / "configs/robot.example.json")
        self.hw["robot"]["left_arm_config"]["cameras"].pop("top")
        with self.assertRaisesRegex(ValueError, "cameras"):
            validate_hardware(self.hw)

    def test_lookahead_cannot_exceed_dynamic_chunk(self):
        self.cfg["deployment"]["stop_window"] = 10
        with self.assertRaisesRegex(ValueError, "minimum"):
            validate_config(self.cfg, True, self.hw)

    def test_legacy_manual_stop_configuration(self):
        self.cfg["deployment"].update(stop_mode="consecutive", stop_window=10, interpolation_multiplier=1)
        self.cfg["deployment"]["dynamic_chunk"]["enabled"] = False
        argv, _ = build_launch(self.cfg, self.hw, True)
        self.assertIn("--stop_signal_mode=consecutive", argv)

    def test_recording_requires_two_leaders(self):
        with self.assertRaisesRegex(ValueError, "teleop"):
            validate_hardware({"robot": self.hw["robot"]}, recording=True)

    def test_home_pose_rejects_missing_or_nonfinite_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            pose = Path(folder) / "home.txt"
            pose.write_text("\n".join(f"{joint}=0" for joint in JOINTS))
            self.assertEqual(len(read_home_pose(pose)), 12)
            pose.write_text(pose.read_text().replace("left_gripper.pos=0", "left_gripper.pos=nan"))
            with self.assertRaises(ValueError):
                read_home_pose(pose)

    def test_checkpoint_action_shape_is_validated_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            self.cfg["deployment"]["checkpoint"] = folder
            (Path(folder) / "config.json").write_text(
                json.dumps({"type": "pi05", "output_features": {"action": {"shape": [12]}}})
            )
            with self.assertRaisesRegex(ValueError, "13"):
                validate_config(self.cfg, True, self.hw, assets=True)


def isolated_definition(path, name):
    """Load one production class/function without importing optional ML drivers."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class ScalarTensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return ScalarTensor(self.value)

    def __add__(self, other):
        return ScalarTensor(self.value + other.value)

    def __sub__(self, other):
        return ScalarTensor(self.value - other.value)

    def __rmul__(self, scalar):
        return ScalarTensor(scalar * self.value)


class VendorCompatibilityTests(unittest.TestCase):
    def test_interpolator_primes_observed_pose_and_supports_boundary_multiplier(self):
        cls = isolated_definition(VENDOR / "policies/rtc/action_interpolator.py", "ActionInterpolator")
        interp = cls(multiplier=3)
        interp.reset(prev=ScalarTensor(10))
        interp.add(ScalarTensor(14), multiplier=4)
        self.assertEqual([interp.get().value for _ in range(4)], [11, 12, 13, 14])
        self.assertTrue(interp.needs_new_action())
        interp.reset(prev=ScalarTensor(100))
        interp.add(ScalarTensor(103))
        self.assertEqual(interp.get().value, 101)

    def test_stop_window_uses_earliest_strict_threshold_window(self):
        path = VENDOR / "common/stop_signal.py"
        spec = importlib.util.spec_from_file_location("stop_signal_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.first_stop_window([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1], 5), 2)
        self.assertIsNone(module.first_stop_window([1, 1, 0.5, 1, 1], 3))
        with self.assertRaises(ValueError):
            module.first_stop_window([float("nan")], 1)

    def test_weight_download_failure_cannot_return_random_policy(self):
        path = VENDOR / "policies/pi05/modeling_pi05.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI05Policy")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
        namespace = {}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        stub = type(
            "Policy",
            (),
            {"from_pretrained": namespace["from_pretrained"], "__init__": lambda self, config, **kwargs: None},
        )
        transformer_utils = types.ModuleType("transformers.utils")
        transformer_utils.cached_file = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("Unavailable checkpoint"))
        with patch.dict(sys.modules, {"transformers.utils": transformer_utils}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Failed to initialize"):
                stub.from_pretrained("missing", config=object(), local_files_only=True)


if __name__ == "__main__":
    unittest.main()
