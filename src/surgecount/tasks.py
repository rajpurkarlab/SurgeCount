"""Canonical dataset task IDs, exact policy prompts, and camera selection.

Dataset IDs are not keyboard digits. Keep these strings identical across
collection, training, and deployment; changing a prompt changes policy input.
"""

TASK_LABELS = {
    0: "gauze_placement",
    1: "return the left arm to rest position",
    2: "release the white edge by opening the left gripper",
    3: "return the right arm to rest position",
    4: "release the white edge and return the left arm to rest position",
    5: "lower and place the gauze into slot 3 with the right arm",
    6: "lower and place the gauze into slot 2 with the right arm",
    7: "release the gauze by opening the right gripper",
    8: "drop that subtask",
    9: "grasp the nearest blue thread of the gauze from the green tray with the right arm and lift it",
    10: "lower and place the gauze into slot 1 with the right arm",
    11: "grasp the exposed blue thread of the gauze with the right arm and lift it",
    12: "grasp the white edge of the gauze with the left arm and lift it",
}

ALL_CAMERAS = (
    "observation.images.left_left_front",
    "observation.images.left_top",
    "observation.images.right_right_front",
)
TASK_CAMERA = {
    1: ALL_CAMERAS[1],
    2: ALL_CAMERAS[0],
    3: ALL_CAMERAS[2],
    4: ALL_CAMERAS[0],
    5: ALL_CAMERAS[1],
    6: ALL_CAMERAS[1],
    7: ALL_CAMERAS[2],
    9: ALL_CAMERAS[2],
    10: ALL_CAMERAS[1],
    11: ALL_CAMERAS[2],
    12: ALL_CAMERAS[0],
}
EXCLUDE_TASKS = frozenset({0, 8})
KEY_TO_TASK = {
    "0": 8,
    "1": 9,
    "2a": 10,
    "2b": 6,
    "2c": 5,
    "3": 7,
    "4": 3,
    "5": 12,
    "6": 11,
    "7": 2,
    "8": 4,
    "9": 1,
}


def validate_task_labels(labels, used_tasks=None):
    """Reject first-occurrence IDs or edited language that would change masking."""
    selected = set(labels) if used_tasks is None else set(map(int, used_tasks))
    for task_id in selected:
        if task_id not in TASK_LABELS:
            raise ValueError(f"Unknown task ID {task_id}; add an explicit prompt and camera mapping first")
        if labels.get(task_id) != TASK_LABELS[task_id]:
            raise ValueError(
                f"Task {task_id} must be {TASK_LABELS[task_id]!r}; got {labels.get(task_id)!r}. "
                "Collection must pre-register the canonical task table."
            )
